"""
Data Service - Handles all data loading, processing, and document creation
Migrated from c_app.py
"""
import os
import re
import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from fuzzywuzzy import process
from langchain_core.documents import Document
from django.conf import settings

# Import from project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SHEET_CONFIG, get_category_mapping, get_column_mapping

logger = logging.getLogger(__name__)

# Global mappings
CATEGORY_MAPPING = None
COLUMN_MAPPING = None


def normalize_colname(name: str) -> str:
    """Normalize column names for consistent matching."""
    name = re.sub(r'[-\s]+', ' ', name.strip().lower())
    name = re.sub(r'\(in\s+sqft\)', '(in sqft)', name)
    return name


def set_mappings_for_type(comparison_type: str) -> None:
    """Set global mappings based on comparison type."""
    global CATEGORY_MAPPING, COLUMN_MAPPING
    cat_map = get_category_mapping(comparison_type)
    col_map = get_column_mapping(comparison_type)
    
    # Normalize column names
    try:
        normalized_col_map = {}
        for key, cols in (col_map or {}).items():
            normalized_cols = [normalize_colname(str(c)) for c in cols]
            normalized_col_map[key] = normalized_cols
    except Exception:
        normalized_col_map = col_map
    
    CATEGORY_MAPPING = cat_map
    COLUMN_MAPPING = normalized_col_map


def get_category_keys(category: str) -> List[str]:
    """Return mapping keys associated with a category."""
    if CATEGORY_MAPPING is None:
        raise RuntimeError("CATEGORY_MAPPING not initialized. Call set_mappings_for_type first.")
    return CATEGORY_MAPPING.get(category.lower(), [])


def get_columns_for_keys(mapping_keys: List[str]) -> Dict[str, List[str]]:
    """Return a dict of mapping_key -> column names filtered by keys."""
    if COLUMN_MAPPING is None:
        raise RuntimeError("COLUMN_MAPPING not initialized. Call set_mappings_for_type first.")
    columns_by_key: Dict[str, List[str]] = {}
    for key in mapping_keys:
        cols = COLUMN_MAPPING.get(key)
        if not cols:
            logger.warning("Mapping key '%s' missing in COLUMN_MAPPING", key)
            continue
        columns_by_key[key] = cols
    return columns_by_key


def flatten_columns(columns_by_key: Dict[str, List[str]]) -> List[str]:
    """Flatten dict of key->columns into a unique column list preserving order."""
    ordered: List[str] = []
    seen = set()
    for cols in columns_by_key.values():
        for col in cols:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
    return ordered


def initialize_dataframe():
    """Initialize dataframe from Excel file."""
    try:
        excel_path = settings.EXCEL_FILE
        pickle_path = settings.PICKLE_FILE
        
        if Path(pickle_path).exists():
            os.remove(pickle_path)
            logger.info("Refreshing data from Excel file...")
        
        if not Path(excel_path).exists():
            logger.error(f"Excel file not found: {excel_path}")
            return None
        
        logger.info(f"Loading data from {excel_path}...")
        dfs = pd.read_excel(excel_path, sheet_name=None)
        logger.info(f"Excel file loaded successfully")
        
        combined = []
        for ctype, cfg in SHEET_CONFIG.items():
            if cfg["sheet"] in dfs:
                df = dfs[cfg["sheet"]].copy()
                df.columns = [normalize_colname(str(c)) for c in df.columns]
                df["__type"] = ctype
                combined.append(df)
        
        if not combined:
            logger.error("No valid sheets found in Excel file!")
            return None
        
        df_all = pd.concat(combined, ignore_index=True)
        joblib.dump(df_all, pickle_path)
        return df_all
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None


def get_comparison_items(comparison_type: str, search_query: Optional[str] = None, limit: int = 100) -> List[str]:
    """Get list of available items for a comparison type, optionally filtered."""
    try:
        pickle_path = settings.PICKLE_FILE
        
        if not Path(pickle_path).exists():
            df_all = initialize_dataframe()
        else:
            df_all = joblib.load(pickle_path)
        
        if df_all is None or df_all.empty:
            logger.error("No data loaded")
            return []
        
        # Filter for comparison type
        type_df = df_all[df_all["__type"] == comparison_type].copy()
        if type_df.empty:
            logger.error(f"No data for comparison type: {comparison_type}")
            return []
        
        # Get items from ID column
        id_col = SHEET_CONFIG[comparison_type]["id_col"]
        
        # Try to find matching column
        if id_col not in type_df.columns:
            cols_norm = {normalize_colname(str(c)): c for c in type_df.columns}
            if id_col in cols_norm:
                matched_col = cols_norm[id_col]
                logger.warning(f"ID column '{id_col}' not present; using normalized match '{matched_col}'")
                id_col = matched_col
            else:
                try:
                    choices = list(cols_norm.keys())
                    best, score = process.extractOne(id_col, choices)
                    if score >= 70:
                        matched_col = cols_norm[best]
                        logger.warning(f"ID column '{id_col}' not found; fuzzy-matched to '{matched_col}' (score={score})")
                        id_col = matched_col
                    else:
                        logger.error(f"ID column '{id_col}' not found for type '{comparison_type}'")
                        return []
                except Exception as exc:
                    logger.exception(f"Error during id_col fuzzy matching: {exc}")
                    return []
        
        unique_items = type_df[id_col].dropna().astype(str).str.strip().str.lower().unique()
        
        # Apply search filter if provided
        if search_query:
            query = search_query.lower().strip()
            # Simple substring match first for speed
            matches = [item for item in unique_items if query in item]
            
            # If few matches, try fuzzy matching (optional, maybe skip for performance on large lists)
            if len(matches) < 5 and len(unique_items) > 0:
                 try:
                    exact_matches = set(matches)
                    candidates = [i for i in unique_items if i not in exact_matches]
                    fuzzy_results = process.extract(query, candidates, limit=10)
                    for item, score in fuzzy_results:
                        if score >= 60:
                            matches.append(item)
                 except Exception:
                     pass
            
            items = sorted(list(set(matches)))
        else:
            items = sorted(unique_items)
            
        # Apply limit
        if limit and limit > 0:
            items = items[:limit]
            
        logger.info(f"Found {len(items)} items for {comparison_type} (query='{search_query}', limit={limit})")
        return items
    
    except Exception as e:
        logger.exception(f"Error getting items for {comparison_type}")
        return []


def load_and_clean_data(
    comparison_type: str,
    items: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    category: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[str]]:
    """
    Load and clean data for analysis.
    
    Returns:
        Tuple of (dataframe, defaults dict, id_column_name)
    """
    try:
        pickle_path = settings.PICKLE_FILE
        
        if Path(pickle_path).exists():
            df = joblib.load(pickle_path)
            df.columns = [normalize_colname(str(c)) for c in df.columns]
            logger.info(f"Pickle file loaded. Shape: {df.shape}")
        else:
            logger.error(f"Pickle file not found at {pickle_path}")
            return None, None, None
        
        # Filter by comparison type
        df = df[df["__type"] == comparison_type].drop(columns=["__type"])
        
        # Resolve ID column
        configured_id = SHEET_CONFIG[comparison_type]["id_col"]
        id_col = configured_id
        if id_col not in df.columns:
            cols_norm = {normalize_colname(str(c)): c for c in df.columns}
            if id_col in cols_norm:
                matched_col = cols_norm[id_col]
                logger.warning(f"ID column '{id_col}' not present; using normalized match '{matched_col}'")
                id_col = matched_col
            else:
                try:
                    choices = list(cols_norm.keys())
                    best, score = process.extractOne(id_col, choices)
                    if score >= 70:
                        matched_col = cols_norm[best]
                        logger.warning(f"ID column '{id_col}' not found; fuzzy-matched to '{matched_col}' (score={score})")
                        id_col = matched_col
                    else:
                        logger.error(f"ID column '{id_col}' not found for type '{comparison_type}'")
                        return None, None, None
                except Exception as exc:
                    logger.exception(f"Error during id_col fuzzy matching: {exc}")
                    return None, None, None
        
        # Clean and normalize ID column values
        try:
            df[id_col] = df[id_col].astype(str).str.strip().str.lower()
        except Exception:
            df[id_col] = df[id_col].astype(str)
        
        available_items = df[id_col].unique()
        logger.info(f"Available {comparison_type}s (sample): {list(available_items)[:20]}")
        
        # Filter by items
        if items:
            lowered = [i.lower() for i in items]
            df = df[df[id_col].isin(lowered)]
            if df.empty:
                # Attempt fuzzy fallback
                try:
                    available = [str(x).strip().lower() for x in available_items]
                    mapped = []
                    mapping_info = {}
                    for orig in lowered:
                        best, score = process.extractOne(orig, available)
                        mapping_info[orig] = (best, score)
                        if score >= 65:
                            mapped.append(best)
                    if mapped:
                        logger.info(f"Fuzzy-mapped requested items {items} -> {mapped}")
                        df = df[df[id_col].isin(mapped)]
                except Exception as exc:
                    logger.warning(f"Fuzzy fallback failed: {exc}")
            
            if df.empty:
                logger.error(f"No data for {comparison_type}s {items}")
                return None, None, None
            logger.info(f"Filtered data for {comparison_type}s {items}. Shape: {df.shape}")
        
        # Year filtering
        if years is None:
            logger.info("No year filtering applied (years=None)")
        else:
            years = [y for y in years if isinstance(y, int) and 1900 <= y <= 9999]
            if years:
                df = df[df["year"].isin(years)]
                logger.info(f"Filtered data for years {years}. Shape: {df.shape}")
        
        # Sort
        sort_cols = [c for c in ["final location", "year"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols)
        
        # Category filtering
        if category and category != "general":
            relevant_columns = ["final location", "year"]
            category_keys = get_category_keys(category)
            category_columns = flatten_columns(get_columns_for_keys(category_keys))
            for col in df.columns:
                if col in category_columns:
                    relevant_columns.append(col)
            relevant_columns = list(dict.fromkeys(relevant_columns))
            df = df[[col for col in relevant_columns if col in df.columns]]
            logger.info(f"Filtered columns for category '{category}'. Shape: {df.shape}")
        
        # Fill defaults
        defaults = {
            "year": 2020,
            "total sold - igr": 0,
            "1bhk_sold - igr": 0,
            "flat total": 0,
            "shop total": 0,
            "office total": 0,
            "others total": 0,
            "1bhk total": 0,
            "<1bhk total": 0
        }
        
        df = df.infer_objects(copy=False).fillna({col: defaults.get(col, 0) for col in df.columns})
        logger.info(f"Final data shape: {df.shape}, columns: {df.columns.tolist()}")
        return df, defaults, id_col
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None


def create_documents(
    df: pd.DataFrame,
    item_ids: List[str],
    defaults: Dict,
    columns_by_key: Dict[str, List[str]],
    years: Optional[List[int]] = None,
    comparison_type: str = "Location",
    id_col: str = "final location"
) -> List[Document]:
    """Create LangChain documents from dataframe for retrieval."""
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]
    
    documents: List[Document] = []
    for mapping_key, columns in columns_by_key.items():
        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            continue
        
        content_lines: List[str] = []
        for item_id in [i.lower() for i in item_ids]:
            item_df = df[df[id_col] == item_id]
            
            # Project-level data (single-row) or no year column
            if comparison_type.strip().lower() == "project" or "year" not in df.columns:
                for col in valid_cols:
                    value = defaults.get(col, 'N/A')
                    if not item_df.empty and col in item_df.columns:
                        try:
                            value = item_df.iloc[0][col]
                        except Exception:
                            value = item_df[col].iloc[0]
                    content_lines.append(f"{item_id}_{mapping_key}_{col}: {value}")
            else:
                # Year-wise data
                for col in valid_cols:
                    year_values = []
                    for year in years:
                        year_df = item_df[item_df["year"] == year]
                        value = year_df[col].iloc[0] if not year_df.empty and col in year_df.columns else defaults.get(col, 'N/A')
                        year_values.append(f"{year}:{value}")
                    content_lines.append(f"{item_id}_{mapping_key}_{col}: {', '.join(year_values)}")
        
        if content_lines:
            documents.append(
                Document(
                    page_content="\n".join(content_lines),
                    metadata={
                        'columns': valid_cols,
                        'items': [i.lower() for i in item_ids],
                        'mapping_key': mapping_key,
                        'years': years,
                    }
                )
            )
            logger.info(f"Created document for mapping key {mapping_key} with columns: {valid_cols}")
    
    logger.info(f"Created {len(documents)} documents for items: {item_ids}")
    return documents


def get_project_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Returns project recommendations with project_name, village, and city."""
    required_cols = ['project name', 'final_location', 'city']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")
    recs = df[required_cols].drop_duplicates().dropna()
    return recs.to_dict(orient='records')


def compute_metrics(
    df: pd.DataFrame,
    mapping_keys: List[str],
    columns_by_key: Dict[str, List[str]],
    item_ids: List[str],
    id_col: str = "final location",
    comparison_type: str = "Location"
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """Compute metrics for items."""
    metrics: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for key in mapping_keys:
        key_columns = [col for col in columns_by_key.get(key, []) if col in df.columns]
        if not key_columns:
            continue
        metrics[key] = {}
        for col in key_columns:
            metrics[key][col] = {}
            for item_id in [i.lower() for i in item_ids]:
                item_df = df[df[id_col] == item_id]
                if item_df.empty:
                    metrics[key][col][item_id] = {
                        "yearly": {},
                        "latest_year": None,
                        "latest_value": None,
                        "total": None,
                        "average": None,
                        "yoy_change": None,
                    }
                    continue
                
                if comparison_type.strip().lower() != "project" and "year" in df.columns:
                    item_df = item_df.sort_values("year")
                    numeric_series = pd.to_numeric(item_df[col], errors='coerce')
                    
                    yearly: Dict[int, float] = {}
                    for _, row in item_df[["year", col]].iterrows():
                        year = int(row["year"]) if pd.notna(row["year"]) else None
                        val = pd.to_numeric(row[col], errors='coerce')
                        yearly[year] = (float(val) if pd.notna(val) else None)
                    
                    latest_year = int(item_df.iloc[-1]["year"]) if pd.notna(item_df.iloc[-1]["year"]) else None
                    latest_val_raw = numeric_series.iloc[-1]
                    latest_value = float(latest_val_raw) if pd.notna(latest_val_raw) else None
                    
                    total_raw = numeric_series.sum(skipna=True)
                    total = float(total_raw) if pd.notna(total_raw) else None
                    
                    avg_raw = numeric_series.mean(skipna=True)
                    average = float(avg_raw) if pd.notna(avg_raw) else None
                    
                    nn = numeric_series.dropna()
                    yoy_change = None
                    if nn.shape[0] >= 2:
                        yoy_change = float(nn.iloc[-1] - nn.iloc[-2])
                    
                    metrics[key][col][item_id] = {
                        "yearly": yearly,
                        "latest_year": latest_year,
                        "latest_value": latest_value,
                        "total": total,
                        "average": average,
                        "yoy_change": yoy_change,
                    }
                else:
                    # Project-level / single-row case
                    val = None
                    if col in item_df.columns:
                        try:
                            raw = item_df.iloc[0][col]
                        except Exception:
                            raw = item_df[col].iloc[0]
                        val = pd.to_numeric(raw, errors='coerce') if pd.notna(raw) else None
                    
                    latest_value = float(val) if pd.notna(val) else None
                    metrics[key][col][item_id] = {
                        "yearly": {},
                        "latest_year": None,
                        "latest_value": latest_value,
                        "total": latest_value,
                        "average": latest_value,
                        "yoy_change": None,
                    }
    return metrics
