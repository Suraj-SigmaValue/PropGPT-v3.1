"""
API Views for PropGPT
"""
import logging
import os
from pathlib import Path
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from django.http import StreamingHttpResponse
import json

# Import serializers
from .serializers import (
    QueryRequestSerializer,
    QueryResponseSerializer,
    ComparisonItemsRequestSerializer,
    ComparisonItemsResponseSerializer,
    FeedbackRequestSerializer,
    FeedbackResponseSerializer,
    CacheStatsResponseSerializer,
    HealthCheckResponseSerializer
)

# Import services
from .services.llm_service import get_llm, count_tokens, clean_response
from .services.data_service import (
    get_comparison_items,
    load_and_clean_data,
    set_mappings_for_type,
    get_category_keys,
    get_columns_for_keys,
    flatten_columns,
    create_documents,
    get_project_recommendations
)
from .services.retrieval_service import (
    get_embeddings,
    build_cache_key,
    build_vector_store,
    build_bm25_retriever,
    hybrid_retrieve
)
from .services.cache_service import (
    get_cached_response,
    set_cached_response,
    delete_cached_response,
    get_cache_stats
)

# Import from project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agents import planner_identify_mapping_keys, agent_pick_relevant_columns, agent_correction_mapping
from prompts import build_location_prompt, build_city_prompt, build_project_prompt
from graph_agent import create_graph

logger = logging.getLogger(__name__)


@api_view(['GET'])
def health_check(request):
    print("Health check endpoint.")
    """Health check endpoint."""
    excel_exists = Path(settings.EXCEL_FILE).exists()
    pickle_exists = Path(settings.PICKLE_FILE).exists()
    
    serializer = HealthCheckResponseSerializer(data={
        'status': 'healthy',
        'version': '1.0.0',
        'data_loaded': pickle_exists,
        'excel_file_exists': excel_exists,
        'pickle_file_exists': pickle_exists
    })
    serializer.is_valid(raise_exception=True)
    return Response(serializer.data)


@api_view(['GET'])
def get_items(request):
    """Get available items for a comparison type."""
    serializer = ComparisonItemsRequestSerializer(data=request.query_params)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    comparison_type = serializer.validated_data['comparison_type']
    search_query = serializer.validated_data.get('search')
    limit = serializer.validated_data.get('limit', 100)
    
    try:
        items = get_comparison_items(comparison_type, search_query, limit)
        response_serializer = ComparisonItemsResponseSerializer(data={
            'comparison_type': comparison_type,
            'items': items,
            'count': len(items)
        })
        response_serializer.is_valid(raise_exception=True)
        return Response(response_serializer.data)
    except Exception as e:
        logger.exception(f"Error getting items: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def process_query(request):
    
    """Main query processing endpoint with streaming support."""
    serializer = QueryRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    stream = data.get('stream', True)
    
    if stream:
        response = StreamingHttpResponse(
            stream_query_generator(data),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'  # Disable Nginx buffering
        return response
    else:
        # Non-streaming implementation (existing logic wrapped in standard response)
        try:
            result = process_query_logic(data)
            response_serializer = QueryResponseSerializer(data=result)
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.data)
        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


def stream_query_generator(data):
    """Generator for streaming query events."""
    try:
        query = data['query']
        comparison_type = data['comparison_type']
        items = data['items']
        categories = data['categories']
        mapping_provider = data.get('mapping_llm_provider', 'openai')
        response_provider = data.get('response_llm_provider', 'openai')
        
        # Send initial status
        yield f"event: status\ndata: Initializing analysis components...\n\n"
        
        # Set mappings
        set_mappings_for_type(comparison_type)
        
        # Initialize LLMs
        mapping_llm = get_llm(mapping_provider)
        response_llm = get_llm(response_provider)
        
        mapping_model = settings.GEMINI_MODEL if mapping_provider == 'gemini' else settings.OPENAI_MODEL
        response_model = settings.GEMINI_MODEL if response_provider == 'gemini' else settings.OPENAI_MODEL
        mapping_provider_display = "Google Gemini" if mapping_provider == 'gemini' else "OpenAI"
        response_provider_display = "Google Gemini" if response_provider == 'gemini' else "OpenAI"
        
        yield f"event: status\ndata: Loading data for {len(items)} items...\n\n"
        
        # Load data
        req_years = None if comparison_type.lower() == "project" else [2020, 2021, 2022, 2023, 2024]
        df, defaults, id_col = load_and_clean_data(
            comparison_type=comparison_type,
            items=items,
            years=req_years
        )
        
        if df is None or df.empty:
            yield f"event: error\ndata: No data found for selected items\n\n"
            return
            
        embeddings = get_embeddings()
        
        # Planning phase
        yield f"event: status\ndata: Planning analysis strategy...\n\n"
        
        forced_keys = data.get('forced_mapping_keys')
        if forced_keys:
             logger.info(f"Using forced mapping keys: {forced_keys}")
             planner_keys = forced_keys
             # We still need to pick columns for these keys
             columns_by_key = get_columns_for_keys(planner_keys)
             candidate_columns = flatten_columns(columns_by_key)
             picked_columns = agent_pick_relevant_columns(mapping_llm, query, planner_keys, candidate_columns)
        else:
            selected_categories = [cat.lower() for cat in categories]
            candidate_keys = []
            for category in selected_categories:
                candidate_keys.extend(get_category_keys(category))
            if not candidate_keys:
                from config import get_column_mapping
                col_map = get_column_mapping(comparison_type)
                candidate_keys = list(col_map.keys())
            candidate_keys = sorted(set(candidate_keys))
            
            try:
                app = create_graph()
                initial_state = {
                    "query": query,
                    "comparison_type": comparison_type,
                    "candidate_keys": candidate_keys,
                    "candidate_columns": [],
                    "llm": mapping_llm,
                    "keys": [],
                    "selected_columns": [],
                    "iteration_count": 0
                }
                import uuid
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                final_state = app.invoke(initial_state, config=config)
                planner_keys = final_state.get("selected_keys", [])
                picked_columns = final_state.get("selected_columns", [])
            except Exception:
                planner_keys = candidate_keys
                picked_columns = []
            
        if not planner_keys: planner_keys = candidate_keys
        columns_by_key = get_columns_for_keys(planner_keys)
        candidate_columns = flatten_columns(columns_by_key)
        if not picked_columns: picked_columns = candidate_columns
        picked_columns = [c for c in picked_columns if c in candidate_columns]
        if not picked_columns: picked_columns = candidate_columns
        
        filtered_columns_by_key = {}
        for key, cols in columns_by_key.items():
            chosen = [col for col in cols if col in picked_columns]
            if chosen: filtered_columns_by_key[key] = chosen
        if not filtered_columns_by_key: filtered_columns_by_key = columns_by_key
        
        final_mapping_keys = list(filtered_columns_by_key.keys())
        final_columns = flatten_columns(filtered_columns_by_key)
        
        yield f"event: status\ndata: Building knowledge context...\n\n"
        
        documents = create_documents(
            df=df,
            item_ids=items,
            defaults=defaults,
            columns_by_key=filtered_columns_by_key,
            years=req_years,
            comparison_type=comparison_type,
            id_col=id_col
        )
        cache_key = build_cache_key(items, final_mapping_keys, final_columns)
        vector_store = build_vector_store(documents, embeddings, cache_key)
        bm25_retriever = build_bm25_retriever(documents)
        
        yield f"event: status\ndata: Retrieving relevant insights...\n\n"
        
        query_context_docs = hybrid_retrieve(query, final_mapping_keys, vector_store, bm25_retriever, top_k=6)
        context = "\n\n".join(doc.page_content.strip() for doc in query_context_docs)
        category_summary = ", ".join(categories)
        
        # Build prompt
        if comparison_type.lower() == "location": build_prompt_func = build_location_prompt
        elif comparison_type.lower() == "city": build_prompt_func = build_city_prompt
        elif comparison_type.lower() == "project": build_prompt_func = build_project_prompt
        else: build_prompt_func = build_location_prompt
        
        formatted_prompt = build_prompt_func(
            question=query.strip(), items=items, mapping_keys=final_mapping_keys,
            selected_columns=final_columns, context=context,
            category_summary=category_summary, chat_history=[]
        )
        
        input_tokens = count_tokens(formatted_prompt, response_model)
        
        # Check cache
        cached_result = get_cached_response(
            query=query.strip(), items=items, mapping_keys=final_mapping_keys,
            comparison_type=comparison_type, provider=response_provider_display,
            embeddings=embeddings
        )
        
        if cached_result:
            response_text, metadata = cached_result
            yield f"event: status\ndata: Cache hit! Streaming cached response...\n\n"
            
            # Simulate streaming for cached response to maintain UI consistency
            chunk_size = 20
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
            
            output_tokens = count_tokens(response_text, response_model)
            if "input_tokens" in metadata: input_tokens = metadata["input_tokens"]
            
            cache_hit = True
        else:
            yield f"event: status\ndata: Generative AI analyzing...\n\n"
            
            full_response = ""
            for chunk in response_llm.stream(formatted_prompt):
                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_response += chunk_text
                yield f"event: token\ndata: {json.dumps(chunk_text)}\n\n"
                
            response_text = full_response
            output_tokens = count_tokens(response_text, response_model)
            
            set_cached_response(
                query=query.strip(), items=items, mapping_keys=final_mapping_keys,
                comparison_type=comparison_type, provider=response_provider_display,
                response=response_text,
                metadata={"input_tokens": input_tokens, "output_tokens": output_tokens, "model": response_model},
                embeddings=embeddings
            )
            cache_hit = False
            
        # Get sheet configuration for data source info
        from config import SHEET_CONFIG
        sheet_info = SHEET_CONFIG.get(comparison_type, {})
        
        # Send final metadata
        metadata = {
            'mapping_keys': final_mapping_keys,
            'selected_columns': final_columns,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'mapping_provider': mapping_provider_display,
            'response_provider': response_provider_display,
            'mapping_model': mapping_model,
            'response_model': response_model,
            'cache_hit': cache_hit,
            'retrieved_sources': [{'content': doc.page_content[:600]} for doc in query_context_docs],
            'data_source': {
                'excel_file': 'Pune_Grand_Summary.xlsx',
                'sheet_name': sheet_info.get('sheet', 'Unknown'),
                'comparison_type': comparison_type,
                'items': items,
                'item_count': len(items)
            }
        }
        
        yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"
        
    except Exception as e:
        logger.exception(f"Stream error: {e}")
        yield f"event: error\ndata: {str(e)}\n\n"

# Helper for non-streaming reuse
def process_query_logic(data):
    # This duplicates logic for non-streaming case - kept simple for this update
    # In production, we'd refactor common logic out. 
    # For now, non-streaming is deprecated but kept for backward compatibility request.
    # We will just call the generator and consume it.
    gen = stream_query_generator(data)
    full_text = ""
    meta = {}
    for event in gen:
        if event.startswith("event: token"):
            text_json = event.split("data: ")[1].strip()
            full_text += json.loads(text_json)
        elif event.startswith("event: metadata"):
            meta_json = event.split("data: ")[1].strip()
            meta = json.loads(meta_json)
    
    return {
        'response': clean_response(full_text),
        **meta
    }


@api_view(['POST'])
def submit_feedback(request):
    """Handle user feedback (thumbs up/down)."""
    serializer = FeedbackRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    data = serializer.validated_data
    feedback_type = data['feedback_type']
    query = data['query']
    items = data['items']
    categories = data['categories']
    old_mapping_keys = data['mapping_keys']
    comparison_type = data['comparison_type']
    provider = data['provider']
    
    try:
        if feedback_type == 'thumbs_down':
            # Delete from cache
            embeddings = get_embeddings()
            deleted = delete_cached_response(
                query=query.strip(),
                items=items,
                mapping_keys=old_mapping_keys,
                comparison_type=comparison_type,
                provider=provider,
                embeddings=embeddings
            )
            
            # Get corrected mapping keys
            set_mappings_for_type(comparison_type)
            mapping_llm = get_llm('openai')
            
            valid_candidates = []
            for cat in categories:
                valid_candidates.extend(get_category_keys(cat.lower()))
            if not valid_candidates:
                from config import get_column_mapping
                col_map = get_column_mapping(comparison_type)
                valid_candidates = list(col_map.keys())
            
            new_keys = agent_correction_mapping(
                mapping_llm,
                query,
                old_mapping_keys,
                sorted(set(valid_candidates))
            )

            # Regenerate response with new keys
            query_data = {
                'query': query,
                'comparison_type': comparison_type,
                'items': items,
                'categories': categories,
                'mapping_llm_provider': provider,
                'response_llm_provider': provider,
                'forced_mapping_keys': new_keys,
                'stream': False
            }
            
            # Use process_query_logic to get the new response
            # Note: This might take a few seconds
            correction_result = process_query_logic(query_data)
            corrected_text = correction_result.get('response', '')
            
            response_data = {
                'status': 'success',
                'message': 'Negative feedback received. Cache cleared and new response generated.',
                'new_mapping_keys': new_keys,
                'corrected_response': corrected_text
            }
        else:
            # Thumbs up - just acknowledge
            response_data = {
                'status': 'success',
                'message': 'Positive feedback received. Thank you!'
            }
        
        response_serializer = FeedbackResponseSerializer(data=response_data)
        response_serializer.is_valid(raise_exception=True)
        return Response(response_serializer.data)
    
    except Exception as e:
        logger.exception(f"Error processing feedback: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def cache_statistics(request):
    """Get cache statistics."""
    try:
        embeddings = get_embeddings()
        stats = get_cache_stats(embeddings)
        
        serializer = CacheStatsResponseSerializer(data=stats)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data)
    except Exception as e:
        logger.exception(f"Error getting cache stats: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['DELETE'])
def clear_cache(request):
    """Clear all cache entries."""
    try:
        embeddings = get_embeddings()
        from .services.cache_service import get_response_cache
        cache = get_response_cache(embeddings)
        cache.cache.clear()
        cache.save_cache()
        
        return Response({
            'status': 'success',
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.exception(f"Error clearing cache: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
