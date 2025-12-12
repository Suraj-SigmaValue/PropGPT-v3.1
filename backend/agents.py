"""
Query Intelligence Agents for PropGPT.

Contains two agents:
1. Planner Agent: Selects relevant mapping keys based on user query
2. Column Agent: Selects relevant columns based on query and mapping keys
"""

import json
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def planner_identify_mapping_keys(llm, query: str, candidate_keys: List[str]) -> List[str]:
    """
    Planner Agent: Identifies and selects the most relevant mapping keys for the user query.
    
    Args:
        llm: Language model instance (LangChain LLM)
        query: User's analysis query
        candidate_keys: List of available mapping keys to choose from
    
    Returns:
        List of selected mapping keys (at most 6, or fewer if fewer are relevant)
    """
    if not candidate_keys:
        return []

    # System instruction - universal semantic understanding
    sys_instr = """You are a data mapping specialist for real-estate analytics.
Your task is to select the MOST RELEVANT mapping keys that will help answer the user's query.
Return ONLY a JSON array of selected mapping keys from the candidate list."""

    prompt = f"""
## USER QUERY
"{query}"

## AVAILABLE MAPPING KEYS
{candidate_keys}

## CORE PRINCIPLE: DIMENSIONAL vs AGGREGATE ANALYSIS

When analyzing a query, you must distinguish between two fundamentally different intents:

1. **AGGREGATE QUERIES**: User wants a single total/summary across all entities
   - Conceptual markers: "total", "overall", "all", "entire"
   - Select keys that return single values or simple lists

2. **DIMENSIONAL QUERIES**: User wants data broken down BY a specific dimension
   - Conceptual markers: Query implies grouping, categorization, or ranking across a dimension
   - Dimensions can be: locations/pincodes, age groups, property types, BHK configs, area ranges, price ranges, etc.
   - Select keys that contain the dimension name AND the metric name together

## SEMANTIC QUERY UNDERSTANDING

### Identify Primary Metric
What measurement does the user want?
- Units (sold/supply)
- Carpet area (consumed/supplied)  
- Sales value (INR)
- Prices/rates
- Project counts
- Shares/percentages

### Identify Analytical Dimension (if any)
Does the query ask to see the metric broken down or grouped by something?
- Geographic (location, pincode)
- Demographic (age, buyer profile)
- Categorical (property type, BHK configuration)
- Ranges (area range, price range)

### Apply Selection Logic
- If NO dimension detected → Select aggregate metric key (e.g., "Total Carpet Area sold")
- If dimension detected → Select dimension-specific breakdown key (e.g., "BHK wise Top 10 Buyer Pincode wise Carpet Area")

### Specificity Hierarchy
Choose the most specific key that matches both the metric AND the dimension:
- Most specific: Keys matching BOTH dimension + metric + sub-category (e.g., BHK + Pincode + Carpet Area)
- Medium: Keys matching dimension + metric (e.g., Property Type + Carpet Area)
- Least: Generic metric keys (e.g., Total Carpet Area)

## CRITICAL DISTINCTIONS

**Supply vs Demand**:
- Supply keywords: available, total units, inventory, planned, capacity, supplied
- Demand keywords: sold, purchased, consumed, transactions, absorbed, bookings

**Composition Analysis**:
- When query asks about mix, share, percentage, distribution → Select keys with "Share (%)"

## TASK
Semantically analyze the query intent. Return 1-6 most relevant mapping keys as a JSON array.
Focus on understanding WHAT the user wants to measure and HOW they want to see it (aggregate vs broken down).

Return ONLY: ["key1", "key2", "key3"]
"""

    try:
        raw_resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw_text = getattr(raw_resp, "content", None) or str(raw_resp)
        
        # Extract JSON array
        start, end = raw_text.find("["), raw_text.rfind("]") + 1
        if start == -1 or end <= 0:
            raise ValueError("Planner did not return JSON array")
        
        parsed = json.loads(raw_text[start:end])
        if not isinstance(parsed, list):
            raise ValueError("Planner output is not a list")
        
        # Filter to only include valid candidate keys
        filtered = [key for key in parsed if key in candidate_keys]

        # Limit to max 6 keys, maintain fallback behavior
        if not filtered:
            return candidate_keys[: min(6, len(candidate_keys))]
        return filtered[:6]


    except Exception as exc:
        logger.warning("[planner_identify_mapping_keys] fallback due to: %s", exc)
        # Simple token-based fallback without hardcoded rules
        q_low = (query or "").lower()
        query_tokens = set(re.findall(r"[\w>]+", q_low))
        
        # Match keys containing query tokens
        heuristic = [
            key for key in candidate_keys
            if any(token in key.lower() for token in query_tokens)
        ]
        
        # Return matches or first 6 candidates as last resort
        return heuristic[:6] if heuristic else candidate_keys[:min(6, len(candidate_keys))]



def agent_pick_relevant_columns(llm, query: str, selected_keys: List[str], candidate_columns: List[str]) -> List[str]:
    """
    Column Agent: Selects the most relevant columns from candidates based on user query and selected keys.
    
    Args:
        llm: Language model instance (LangChain LLM)
        query: User's analysis query
        selected_keys: List of selected mapping keys (from planner agent)
        candidate_columns: List of available columns to choose from
    
    Returns:
        List of selected column names (typically 5-20 relevant columns)
    """
    if not candidate_columns:
        return []

    sys_instr = (
        "You select strictly relevant dataframe column names for the user's analytics query. "
        "Return ONLY a JSON list of exact column names from CANDIDATE_COLUMNS—no extra text."
    )
    prompt = f"""
    User Query: {query}

    Selected Mapping Keys (context only):
    {json.dumps(selected_keys, indent=2)}

    CANDIDATE_COLUMNS:
    {json.dumps(candidate_columns, indent=2)}

    Rules:
    - Choose only columns that are directly useful to answer the query (avoid generic/noise columns).
    - Keep the set small but sufficient (usually 5–20).
    - Output ONLY a JSON array of column names from CANDIDATE_COLUMNS. No markdown, no commentary.
    - CRITICAL: You MUST select at least one column for EVERY mapping key in "Selected Mapping Keys". Do not ignore any mapping key.
    - If user asked question about "Demand" refer unit sold mapping keys if no specific mention refer Property type wise unit sold mapping keys
    - If a mapping key seems to have multiple relevant columns, pick the most descriptive ones.
    """
    try:
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        s, e = raw.find("["), raw.rfind("]") + 1
        if s == -1 or e <= 0:
            raise ValueError("Agent did not return a JSON list.")
        picked = json.loads(raw[s:e])
        if not isinstance(picked, list):
            raise ValueError("Agent output is not a list.")
        picked = [c for c in picked if c in candidate_columns]
        picked = list(dict.fromkeys(picked))
        return picked or candidate_columns[: min(15, len(candidate_columns))]
    except Exception as exc:
        logger.warning("[agent_pick_relevant_columns] fallback due to: %s", exc)
        query_tokens = re.findall(r"\w+", query.lower())
        heuristic = [
            col for col in candidate_columns
            if any(token in col.lower() for token in query_tokens)
        ]
        return heuristic or candidate_columns[: min(15, len(candidate_columns))]


def agent_correction_mapping(llm, query: str, old_keys: List[str], candidate_keys: List[str]) -> List[str]:
    """
    Correction Agent: Proposes NEW mapping keys assuming the old ones were incorrect (Thumbs Down).
    
    Args:
        llm: Language model instance
        query: User's original query
        old_keys: The keys used in the rejected response
        candidate_keys: All available keys
        
    Returns:
        New list of mapping keys
    """
    if not candidate_keys:
        return []

    sys_instr = (
        "You are a correction assistant. The user provided negative feedback (Thumbs Down) for a previous answer. "
        "The previous answer used a specific set of mapping keys which the user ostensibly found incorrect or insufficient. "
        "Your task: Re-analyze the query and select BETTER mapping keys from CANDIDATE_KEYS. "
        "Avoid simply repeating the exact same set if possible, unless you are strictly convinced they are the only correct ones "
        "(in which case, maybe add a missing key). "
        "Return ONLY a JSON list of mapping keys."
    )
    
    prompt = f"""
    ### SYSTEM ROLE: REINFORCEMENT LEARNING CORRECTION AGENT
    You are an intelligent data mapping agent. You are currently in a "Correction Loop" because your previous action received a NEGATIVE REWARD (User Thumbs Down).

    ### CURRENT STATE
    1. **User Query:** "{query}"
    2. **Rejected Policy (Previous Incorrect Keys):** {json.dumps(old_keys, indent=2)}

    ### ACTION SPACE (Available Candidate Keys)
    {json.dumps(candidate_keys, indent=2)}

    ### OPTIMIZATION TASK
    Your goal is to maximize the reward by finding the correct mapping that the user accepts.
    You must apply "Reflexion" to diagnose the error and switch your strategy.

    **Step 1: Diagnostic (Critique Phase)**
    Analyze WHY the Rejected Keys resulted in a negative reward.
    - Did you confuse Supply (Total Units) vs Demand (Sold)?
    - Did you confuse Granularity (Daily vs Monthly)?
    - Did you miss specific metadata filters (e.g., specific region or status)?
    *Note: The user implies the previous mapping was logically inverted or irrelevant.*

    **Step 2: Exploration (Correction Phase)**
    Select a DIFFERENT set of keys from the Candidate List that satisfies the query. 
    - CONSTRAINT: You MUST NOT output the exact same set of keys as the Rejected Policy.
    - HEURISTIC: If the query implies "Anti-Gravity" or high-level abstraction, look for computed columns or parent categories.

    ### OUTPUT FORMAT
    Provide your response in this strict JSON format only, with no markdown code blocks:

    {{
    "reasoning_trace": "Brief explanation of why the old keys failed and why the new ones were chosen.",
    "corrected_keys": ["key1", "key2"]
    }}
    """
    
    try:
        resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw = getattr(resp, "content", None) or str(resp)
        s, e = raw.find("["), raw.rfind("]") + 1
        if s == -1 or e <= 0:
            # Fallback: Just return the old keys if parsing fails, or try heuristic
            return old_keys
        
        parsed = json.loads(raw[s:e])
        if not isinstance(parsed, list):
            return old_keys
            
        filtered = [k for k in parsed if k in candidate_keys]
        if not filtered:
             # If agent went rogue and returned invalid keys, fallback to old keys or top candidates
             return candidate_keys[:3]
             
        return filtered
        
    except Exception as e:
        logger.warning(f"Correction agent failed: {e}")
        return old_keys
