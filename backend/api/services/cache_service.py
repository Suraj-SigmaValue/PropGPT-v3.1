"""
Cache Service - Wrapper for semantic response cache
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from django.conf import settings

# Import response cache from project root
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from response_cache import SemanticResponseCache

logger = logging.getLogger(__name__)

# Global cache instance
_cache_instance = None


def get_response_cache(embeddings):
    """Get or create semantic response cache instance."""
    global _cache_instance
    if _cache_instance is None:
        cache_dir = Path(settings.RESPONSE_CACHE_DIR)
        _cache_instance = SemanticResponseCache(
            cache_dir=cache_dir,
            embeddings=embeddings,
            similarity_threshold=0.95,
            ttl_seconds=86400
        )
    return _cache_instance


def get_cached_response(
    query: str,
    items: List[str],
    mapping_keys: List[str],
    comparison_type: str,
    provider: str,
    embeddings
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Get cached response if available.
    
    Returns:
        Tuple of (response_text, metadata) or None if not found
    """
    cache = get_response_cache(embeddings)
    return cache.get(
        query=query.strip(),
        items=items,
        mapping_keys=mapping_keys,
        comparison_type=comparison_type,
        provider=provider
    )


def set_cached_response(
    query: str,
    items: List[str],
    mapping_keys: List[str],
    comparison_type: str,
    provider: str,
    response: str,
    metadata: Dict[str, Any],
    embeddings
):
    """Store response in cache."""
    cache = get_response_cache(embeddings)
    cache.set(
        query=query.strip(),
        items=items,
        mapping_keys=mapping_keys,
        comparison_type=comparison_type,
        provider=provider,
        response=response,
        metadata=metadata
    )


def delete_cached_response(
    query: str,
    items: List[str],
    mapping_keys: List[str],
    comparison_type: str,
    provider: str,
    embeddings
) -> bool:
    """Delete cached response."""
    cache = get_response_cache(embeddings)
    return cache.delete(
        query=query.strip(),
        items=items,
        mapping_keys=mapping_keys,
        comparison_type=comparison_type,
        provider=provider
    )


def get_cache_stats(embeddings) -> Dict[str, Any]:
    """Get cache statistics."""
    cache = get_response_cache(embeddings)
    return {
        "total_entries": len(cache.cache),
        "cache_dir": str(cache.cache_dir),
        "similarity_threshold": cache.similarity_threshold,
        "ttl_seconds": cache.ttl_seconds
    }
