"""
Retrieval Service - Handles vector store, BM25, and hybrid retrieval
"""
import logging
from hashlib import md5
import json
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from django.conf import settings

logger = logging.getLogger(__name__)


def get_embeddings():
    """Get HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={'normalize_embeddings': True}
    )


def build_cache_key(items: List[str], mapping_keys: List[str], columns: List[str]) -> str:
    """Build cache key for vector store."""
    payload = {
        "items": sorted([i.lower() for i in items]),
        "keys": sorted(mapping_keys),
        "columns": sorted(columns),
    }
    return md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_vector_store(documents: List[Document], embeddings: HuggingFaceEmbeddings, cache_key: str):
    """Build or load FAISS vector store."""
    index_dir = Path(settings.VECTOR_CACHE_DIR)
    index_dir.mkdir(exist_ok=True)
    vector_store_path = index_dir / cache_key
    
    if vector_store_path.exists():
        try:
            store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded FAISS index from {vector_store_path}")
            return store
        except Exception as exc:
            logger.warning(f"Failed to load FAISS index ({exc}). Rebuilding.")
    
    store = FAISS.from_documents(documents, embeddings)
    store.save_local(str(vector_store_path))
    logger.info(f"FAISS index saved to {vector_store_path}")
    return store


def build_bm25_retriever(documents: List[Document]):
    """Build BM25 retriever."""
    try:
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = 8
        return retriever
    except Exception as exc:
        logger.warning(f"Failed to initialize BM25 retriever: {exc}")
        return None


def hybrid_retrieve(query: str, mapping_keys: List[str], vector_store: FAISS, bm25_retriever, top_k: int = 6):
    """Perform hybrid retrieval using both FAISS and BM25."""
    retrieved = []
    seen_contents = set()
    
    for key in mapping_keys:
        faiss_docs = []
        if vector_store:
            try:
                faiss_docs = vector_store.similarity_search(query, k=top_k, filter={"mapping_key": key})
            except Exception as exc:
                logger.warning(f"FAISS search failed for key {key}: {exc}")
        
        bm25_docs = []
        if bm25_retriever:
            try:
                # CORRECTED: Use invoke() method instead of _get_relevant_documents()
                bm25_results = bm25_retriever.invoke(query)
                bm25_docs = [doc for doc in bm25_results if doc.metadata.get("mapping_key") == key][:top_k]
            except Exception as exc:
                logger.warning(f"BM25 retrieval failed for key {key}: {exc}")
        
        combined = faiss_docs + bm25_docs
        for doc in combined:
            content_hash = md5(doc.page_content.encode()).hexdigest()
            if content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)
            retrieved.append(doc)
    
    logger.info(f"Hybrid retrieval returned {len(retrieved)} documents")
    return retrieved
