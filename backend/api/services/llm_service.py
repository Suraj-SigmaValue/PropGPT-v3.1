"""
LLM Service - Handles LLM initialization and query processing
"""
import os
import logging
import tiktoken
from typing import Optional
from django.conf import settings

logger = logging.getLogger(__name__)


def get_llm(provider_name: Optional[str] = None):
    """
    Initialize and return LLM instance based on provider.
    
    Args:
        provider_name: 'openai' or 'gemini'. If None, uses settings.USE_LLM
    
    Returns:
        LLM instance (ChatOpenAI or ChatGoogleGenerativeAI)
    """
    from langchain_openai import ChatOpenAI
    
    # Determine provider
    provider = (provider_name or settings.USE_LLM).strip().lower()
    logger.info(f"Using LLM provider: {provider}")
    
    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-google-genai not installed. "
                "Run: pip install langchain-google-genai google-generativeai"
            ) from exc
        
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY for Gemini.")
        
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.2,
            max_output_tokens=8192,
            convert_system_message_to_human=True,
        )
    
    # Default to OpenAI
    api_key = settings.OPENAI_API_KEY
    if not api_key or not api_key.startswith("sk-"):
        raise RuntimeError("Missing/invalid OPENAI_API_KEY.")
    
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=api_key,
        temperature=0.3,
        max_completion_tokens=15000,
        max_retries=3,
    )


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for encoding
    
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0


def clean_response(text: str) -> str:
    """
    Clean and format LLM response while preserving markdown structure.
    
    Args:
        text: Raw LLM response text
    
    Returns:
        Cleaned and formatted text
    """
    import re
    
    if not text:
        return ""
    
    # Remove markdown code block markers
    text = re.sub(r'^```markdown\s*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n```\s*$', '', text)
    
    # Fix broken words before headers
    text = re.sub(r'(\w+)\s*\n+\s*(#{1,6}\s+)', r'\1 \2', text)
    
    # Ensure proper spacing before headers WITHOUT breaking words
    text = re.sub(r'([^\n])\s*(#{1,6}\s+)', r'\1\n\n\2', text)
    
    # Ensure proper spacing after headers
    text = re.sub(r'(#{1,6}\s+.+?)\s*([^\n])', r'\1\n\2', text, flags=re.DOTALL)
    
    # Fix broken words before bullet points
    text = re.sub(r'(\w+)\s*\n+\s*([-*]\s+)', r'\1 \2', text)
    
    # Ensure newlines before bullet points
    text = re.sub(r'([^\n])\s*([-*]\s+)', r'\1\n\2', text)
    
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Repair remaining broken words
    text = re.sub(r'(\w+)\\n(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)\n(\w+)', r'\1\2', text)
    
    return text.strip()
