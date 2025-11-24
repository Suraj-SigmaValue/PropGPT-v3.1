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

    sys_instr = (
        "You are a planning assistant that selects the most relevant mapping keys for answering a "
        "real-estate analytics question. Return ONLY a JSON list of mapping keys from CANDIDATE_KEYS."
    )
    prompt = f"""
    User Query: {query}

    CANDIDATE_KEYS:
    {json.dumps(candidate_keys, indent=2)}

    Rules:
    - Break a multi-metric query into its individual metric components.
    - For every component, return the most specific mapping key available.
    - If no mapping key exists for that component, return a placeholder key using the user’s same wording.  
    - Do not omit or merge components.
    - Ensure the final mapping covers 100% of the metrics mentioned by the user.
    - Never return any mapping key that contains additional qualifiers, dimensions, or sub-metrics unless those qualifiers appear in the user query.
    - if user asked question about "Demand" refer unit sold mapping keys
    - if user asked question related to "Supply" refer Total unit mapping keys
    - if no specific Segment/Property type/BHK mention in user query refer Property type wise mapping keys
    - if user asked for sales value, refer to mapping keys related to total sales (INR)
    - if user ask regarding number of projects launched, phases launched, and total buildings or towers refer "Total Project Launched","Total Phases Launched " and "Total Buildings or Towers" Mapping.
    - if user asked question related to project, select mapping project related keys
    - Choose the smallest set of mapping keys that covers the metrics implied by the question.
    - Prefer specific keys over broad ones.
    - Output ONLY a JSON array (no commentary).
    - if user ask for anything regarding Carpet area consumed by top pin code in >3 BHK. refer to mapping keys BHK type wise top buyers pincode wise Carpet area sold or consumed Percentage(%)
    - if user ask for "stock and complexity snapshot" or similar, ensure you select ALL relevant keys: "Total Project Launched", "Total Phases Launched", and "Total Buildings or Towers".
    """
    try:
        raw_resp = llm.invoke(sys_instr + "\n\n" + prompt)
        raw_text = getattr(raw_resp, "content", None) or str(raw_resp)
        start, end = raw_text.find("["), raw_text.rfind("]") + 1
        if start == -1 or end <= 0:
            raise ValueError("Planner did not return JSON array")
        parsed = json.loads(raw_text[start:end])
        if not isinstance(parsed, list):
            raise ValueError("Planner output is not a list")
        filtered = [key for key in parsed if key in candidate_keys]
        return filtered or candidate_keys[: min(6, len(candidate_keys))]
    except Exception as exc:
        logger.warning("[planner_identify_mapping_keys] fallback due to: %s", exc)
        query_tokens = set(re.findall(r"[\w>]+", query.lower()))
        heuristic = [
            key for key in candidate_keys
            if any(token in key.lower() for token in query_tokens)
        ]
        return heuristic or candidate_keys[: min(6, len(candidate_keys))]


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
    -if user asked question about "Demand" refer unit sold mapping keys if no specific mention refer Property type wise unit sold mapping keys
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
