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

    # -------------------------
    # Local helper: classify query by type (BHK config etc.)
    # -------------------------
    def _classify_query_type(q: str) -> str:
        q = (q or "").lower()

        # BHK / configuration demand-style queries
        bhk_tokens = [
            "configuration", "configurations", "config mix",
            "smaller units", "larger units",
            "smaller configuration", "larger configuration",
            "small size", "big size", "bigger units",
            "1bhk", "2bhk", "3bhk", "4bhk", "bhk",
        ]
        if any(tok in q for tok in bhk_tokens):
            return "bhk_config"

        return "generic"

    # -------------------------
    # Local helper: classify metric as supply / demand / both
    # -------------------------
    def _classify_metric(q: str) -> str:
        q = (q or "").lower()

        supply_tokens = [
            "supply", "available", "inventory", "stock",
            "total units", "unsold", "launched", "supplied",
        ]
        demand_tokens = [
            "demand", "sold", "purchased", "bought",
            "transactions", "sales count", "absorbed", "consumed",
        ]

        has_supply = any(tok in q for tok in supply_tokens)
        has_demand = any(tok in q for tok in demand_tokens)

        if has_supply and has_demand:
            return "both"
        if has_supply:
            return "supply"
        if has_demand:
            return "demand"
        return "unknown"

    # -------------------------
    # Local helper: detect demography / pincode analysis queries
    # -------------------------
    def _is_demography_query(q: str) -> bool:
        q = (q or "").lower()
        demo_tokens = [
            "demography", "demographic", "demographics",
            "buyer profile", "buyer mix", "age profile",
            "pincode", "pin code", "area-wise buyers", "buyer location",
            "top 10 buyer", "top buyer area", "top buyer pincode",
        ]
        return any(tok in q for tok in demo_tokens)

    # -------------------------
    # Local helper: detect property type and BHK mentions
    # -------------------------
    def _detect_property_mentions(q: str) -> dict:
        q = (q or "").lower()
        return {
            "has_property_type": any(tok in q for tok in ["flat", "shop", "office", "property type", "property"]),
            "has_bhk_type": any(tok in q for tok in ["1bhk", "2bhk", "3bhk", "4bhk", "bhk", "bedroom", "configuration"])
        }

    q_low = (query or "").lower()
    q_type = _classify_query_type(query)
    metric_type = _classify_metric(query)
    is_demo = _is_demography_query(query)
    property_mentions = _detect_property_mentions(query)

    sys_instr = (
        "You are a planning assistant that selects the most relevant mapping keys for answering a "
        "real-estate analytics question. Return ONLY a JSON list of mapping keys from CANDIDATE_KEYS."
    )
    prompt = f"""
    User Query: {query}

    CANDIDATE_KEYS:
    {json.dumps(candidate_keys, indent=2)}

    Selection Rules (in priority order):
    
    CRITICAL - SOLD vs TOTAL UNITS DISTINCTION:
    - "SOLD" / "DEMAND" queries → Use "Property type wise Units Sold" (contains sold data from IGR)
    - "TOTAL UNITS" / "SUPPLY" queries → Use "Property Type wise total units" (contains total available units)
    - Keywords for SOLD: "sold", "demand", "purchased", "bought", "transactions", "sales count"
    - Keywords for TOTAL: "supply", "available", "total units", "inventory", "stock"
    - DEFAULT: If query asks about flats/shops/offices WITHOUT specifying supply/total → Assume SOLD units
    
    1. SUPPLY & DEMAND QUERIES:
       - If query mentions BOTH "supply" AND "demand": Include "Property type wise Units Sold" (demand) + "Property Type wise total units" (supply)
       - If query mentions only "supply" or "total units" or "available": 
         - No property type mentioned → Select "total units"
         - Property type mentioned → Select "Property Type wise total units" 
         - BHK type mentioned → Select "BHK wise total units"
       - If query mentions only "demand" or "sold": Select "Property type wise Units Sold", "BHK types wise units sold"
       - If query asks "how many flats/shops/offices" WITHOUT "supply"/"total" keywords → Select "Property type wise Units Sold"
       - If query asks related to rate trend refer Property Type Wise Average Price per Sq. Ft. (Carpet Area Basis) mapping
    
    2. SPECIFIC METRICS:
       - Sales value (INR) → "Total sales (INR)" and property/BHK-specific sales keys
       - Project count → "Total Project Launched", "Total Phases Launched", "Total Buildings or Towers"
       - Carpet area by pincode in >3BHK → "BHK type wise top buyers pincode wise Carpet area sold or consumed Percentage(%)"
    
    3. PROPERTY TYPE & BHK EXPANSION:
       - If NO specific property type/BHK mentioned → Include property-type-wise keys
       - If specific type mentioned → Include only that type's keys
    
    4. GENERAL SELECTION PRINCIPLES:
       - Return the smallest set of keys that fully answers the query
       - Match user's specificity level (don't add qualifiers not in query)
       - Break multi-metric queries into components and cover ALL components
    
    EXAMPLES:
    - "How many flats sold?" → ["Property type wise Units Sold"]
    - "Total units available?" → ["total units"]
    - "Supply of flats?" → ["Property Type wise total units"]
    - "Supply of 2BHK units?" → ["BHK wise total units"]
    - "Flats, shops, offices sold in each location?" → ["Property type wise Units Sold", "Location"]
    
    Output: JSON array of mapping keys only, no commentary.
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

        # -------------------------
        # Deterministic hard rules (do NOT rely only on LLM)
        # -------------------------

        # 1) If query is about "location", ensure "Location" mapping key is present
        if "location" in q_low and "Location" in candidate_keys and "Location" not in filtered:
            filtered.insert(0, "Location")

        # 2) If query is BHK/configuration-style, ensure BHK demand key is present
        if q_type == "bhk_config":
            if "BHK types wise units sold" in candidate_keys and "BHK types wise units sold" not in filtered:
                filtered.append("BHK types wise units sold")

        # 3) SUPPLY vs DEMAND ENFORCEMENT with granular property/BHK detection
        if metric_type in ("supply", "both"):
            # Priority order for supply keys based on query specificity
            supply_key_priority = []
            
            if property_mentions["has_bhk_type"]:
                # BHK-specific supply query
                supply_key_priority.extend([
                    "BHK wise total units",
                    "BHK wise total carpet area in sqft",
                    "Property Type wise total units", 
                    "total units"
                ])
            elif property_mentions["has_property_type"]:
                # Property-type-specific supply query
                supply_key_priority.extend([
                    "Property Type wise total units",
                    "Property type wise Total Carpet Area (in sq ft)",
                    "total units"
                ])
            else:
                # General supply query
                supply_key_priority.extend([
                    "total units",
                    "Total Carpet Area (In sq ft)",
                    "Property Type wise total units",
                    "BHK wise total units"
                ])
            
            # Add priority supply keys that are available
            for key in supply_key_priority:
                if key in candidate_keys and key not in filtered:
                    filtered.append(key)

            # Push sold/sales keys to the end for supply queries
            demand_like = []
            non_demand_like = []
            for k in filtered:
                if ("units sold" in k.lower()) or ("sold" in k.lower()) or ("total sales" in k.lower()):
                    demand_like.append(k)
                else:
                    non_demand_like.append(k)
            filtered = non_demand_like + demand_like

        # 4) DEMOGRAPHY / PINCODE ENFORCEMENT
        if is_demo:
            demo_key = "Top 10 Buyer Pincode units sold"
            if demo_key in candidate_keys and demo_key not in filtered:
                filtered.append(demo_key)

        # Limit to max 6 keys, keep existing behaviour if list is empty
        if not filtered:
            return candidate_keys[: min(6, len(candidate_keys))]
        return filtered[:6]

    except Exception as exc:
        logger.warning("[planner_identify_mapping_keys] fallback due to: %s", exc)
        q_low = (query or "").lower()
        query_tokens = set(re.findall(r"[\w>]+", q_low))
        heuristic = [
            key for key in candidate_keys
            if any(token in key.lower() for token in query_tokens)
        ]

        filtered = heuristic or candidate_keys[: min(6, len(candidate_keys))]

        # Same deterministic guarantees in fallback

        if "location" in q_low and "Location" in candidate_keys and "Location" not in filtered:
            filtered.insert(0, "Location")

        if _classify_query_type(query) == "bhk_config":
            if "BHK types wise units sold" in candidate_keys and "BHK types wise units sold" not in filtered:
                filtered.append("BHK types wise units sold")

        # Enhanced supply handling in fallback
        metric_type_fallback = _classify_metric(query)
        property_mentions_fallback = _detect_property_mentions(query)
        
        if metric_type_fallback in ("supply", "both"):
            supply_key_priority = []
            
            if property_mentions_fallback["has_bhk_type"]:
                supply_key_priority.extend([
                    "BHK wise total units",
                    "BHK wise total carpet area in sqft",
                    "Property Type wise total units", 
                    "total units"
                ])
            elif property_mentions_fallback["has_property_type"]:
                supply_key_priority.extend([
                    "Property Type wise total units",
                    "Property type wise Total Carpet Area (in sq ft)",
                    "total units"
                ])
            else:
                supply_key_priority.extend([
                    "total units",
                    "Total Carpet Area (In sq ft)",
                    "Property Type wise total units",
                    "BHK wise total units"
                ])
            
            for key in supply_key_priority:
                if key in candidate_keys and key not in filtered:
                    filtered.append(key)

            demand_like = []
            non_demand_like = []
            for k in filtered:
                if ("units sold" in k.lower()) or ("sold" in k.lower()) or ("total sales" in k.lower()):
                    demand_like.append(k)
                else:
                    non_demand_like.append(k)
            filtered = non_demand_like + demand_like

        if _is_demography_query(query):
            demo_key = "Top 10 Buyer Pincode units sold"
            if demo_key in candidate_keys and demo_key not in filtered:
                filtered.append(demo_key)

        return filtered[:6]


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