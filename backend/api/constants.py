"""
Centralized constants for PropGPT API.
This file contains all hardcoded values that were scattered across the codebase.
"""

# API Constants
DEFAULT_COMPARISON_TYPES = ['Location', 'City', 'Project']
DEFAULT_LLM_PROVIDERS = ['openai', 'gemini']
DEFAULT_FEEDBACK_TYPES = ['thumbs_up', 'thumbs_down']

# Default values for query processing
DEFAULT_CATEGORIES = ['all', 'demand', 'supply', 'price', 'demography']
DEFAULT_LLM_PROVIDER = 'openai'
DEFAULT_STREAM = True
DEFAULT_ITEM_LIMIT = 100

# Field constraints
MAX_QUERY_LENGTH = 5000
MAX_ITEM_NAME_LENGTH = 200
MAX_CATEGORY_NAME_LENGTH = 100
MIN_ITEMS_COUNT = 1
MAX_ITEMS_COUNT = 5

# Pagination
DEFAULT_PAGE_SIZE = 100

# Token counting defaults
DEFAULT_TOKEN_MODEL = 'gpt-4o-mini'
