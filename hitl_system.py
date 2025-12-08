import streamlit as st
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def on_thumbs_up():
    """Callback for positive feedback."""
    st.session_state.feedback_given = True
    st.session_state.feedback_type = "up"
    # Log analytics here if needed
    logger.info("User provided positive feedback (Thumbs Up).")
    st.toast("Thank you for your feedback! 👍")

def on_thumbs_down():
    """Callback for negative feedback."""
    st.session_state.feedback_given = True
    st.session_state.feedback_type = "down"
    st.session_state.needs_retry = True
    logger.info("User provided negative feedback (Thumbs Down). Triggering retry.")
    st.toast("Feedback received. Recalculating... 👎")

def show_feedback_ui(query, items, categories, mapping_keys, df_snapshot_info):
    """
    Renders the Thumbs Up / Thumbs Down UI.
    Stores context in session_state for potential retry.
    
    Args:
        query: User query string
        items: Selected items list
        categories: Selected categories list
        mapping_keys: The mapping keys used in the response
        df_snapshot_info: basic info to identify data state (e.g. comparison_type)
    """
    # Only show if feedback hasn't been given for this turn
    # We use a unique key for the buttons based on the query to reset state for new queries
    
    # Store context immediately, so it's available if 'down' is clicked
    # We rely on persistent session state across reruns
    st.session_state.last_run_context = {
        "query": query,
        "items": items,
        "categories": categories,
        "old_mapping_keys": mapping_keys,
        "df_info": df_snapshot_info
    }

    st.markdown("---")
    st.write("Please rate this answer:")
    
    col1, col2, col3 = st.columns([1, 1, 10])
    
    with col1:
        st.button("👍", key=f"thumbs_up_{hash(query)}", on_click=on_thumbs_up, help="Accurate and helpful")
        
    with col2:
        st.button("👎", key=f"thumbs_down_{hash(query)}", on_click=on_thumbs_down, help="Inaccurate or wrong mapping")

