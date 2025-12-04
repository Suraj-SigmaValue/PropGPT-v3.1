import json
import logging
from typing import TypedDict, List, Annotated, Dict, Any, Literal
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from IPython.display import Image, display

from agents import planner_identify_mapping_keys, agent_pick_relevant_columns
from mapping import (
    COLUMN_MAPPING_Location,
    COLUMN_MAPPING_Project,
    COLUMN_MAPPING_City
)

# Configure logging
logger = logging.getLogger(__name__)

# --- State Definition ---
class AgentState(TypedDict):
    query: str
    llm: Any
    comparison_type: str
    candidate_keys: List[str]
    candidate_columns: List[str]
    selected_keys: List[str]
    selected_columns: List[str]
    messages: Annotated[List[BaseMessage], operator.add]
    iteration_count: int  # To prevent infinite loops
    final_response: str  # LLM generated response

# --- Tools (Intelligent Layer) ---

@tool
def identify_keys_tool(query: str, candidate_keys: List[str], llm: Any, messages: List[Any] = None) -> Dict[str, Any]:
    """
    Identifies relevant mapping keys based on the query.
    """
    logger.info("--- Tool: Identify Keys ---")
    
    # Extract text history from messages
    history = []
    if messages:
        for msg in messages:
            if hasattr(msg, 'content'):
                history.append(f"{type(msg).__name__}: {msg.content}")
            else:
                history.append(str(msg))

    selected_keys = planner_identify_mapping_keys(llm, query, candidate_keys)
    logger.info(f"Selected Keys: {selected_keys}")
    return {"selected_keys": selected_keys}

@tool
def pick_columns_tool(query: str, selected_keys: List[str], candidate_columns: List[str], llm: Any) -> Dict[str, Any]:
    """
    Picks relevant columns based on selected keys and query.
    """
    logger.info("--- Tool: Pick Columns ---")
    
    selected_columns = agent_pick_relevant_columns(llm, query, selected_keys, candidate_columns)
    logger.info(f"Selected Columns: {selected_columns}")
    return {"selected_columns": selected_columns}

# --- Nodes (Mapping Layers & Flow) ---

def load_mappings_node(state: AgentState):
    """
    Node to load the appropriate mappings based on comparison type.
    This acts as the 'mapping layer' node.
    """
    logger.info("--- Node: Load Mappings ---")
    
    # If candidate_keys are already provided (e.g. filtered by category in UI), preserve them
    if state.get("candidate_keys"):
        logger.info("Using provided candidate_keys (filtered by category)")
        return {
            "candidate_keys": state["candidate_keys"],
            # If candidate_columns not provided, we might need to load them? 
            # Usually c_app.py doesn't pass candidate_columns initially, it generates them after keys are selected.
            # But the graph defines 'candidate_columns' in state. 
            # Let's load ALL columns as candidates if not provided, or just leave empty?
            # The column picker needs candidate_columns.
            "candidate_columns": state.get("candidate_columns", []),
            "iteration_count": 0
        }

    comparison_type = state.get("comparison_type", "Location")
    
    candidate_keys = []
    candidate_columns = []
    
    if comparison_type == "Location":
        candidate_keys = list(COLUMN_MAPPING_Location.keys())
        all_cols = []
        for cols in COLUMN_MAPPING_Location.values():
            all_cols.extend(cols)
        candidate_columns = list(set(all_cols))
        
    elif comparison_type == "Project":
        candidate_keys = list(COLUMN_MAPPING_Project.keys())
        all_cols = []
        for cols in COLUMN_MAPPING_Project.values():
            all_cols.extend(cols)
        candidate_columns = list(set(all_cols))
        
    elif comparison_type == "City":
        candidate_keys = list(COLUMN_MAPPING_City.keys())
        all_cols = []
        for cols in COLUMN_MAPPING_City.values():
            all_cols.extend(cols)
        candidate_columns = list(set(all_cols))
        
    else:
        candidate_keys = []
        candidate_columns = []

    return {
        "candidate_keys": candidate_keys,
        "candidate_columns": candidate_columns,
        "iteration_count": 0
    }

def planner_node(state: AgentState):
    """
    Node that executes the identify_keys_tool logic.
    """
    result = identify_keys_tool.invoke({
        "query": state["query"],
        "candidate_keys": state["candidate_keys"],
        "llm": state["llm"],
        "messages": state.get("messages", [])
    })
    
    # Increment iteration count
    return {
        **result,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def column_node(state: AgentState):
    """
    Node that executes the pick_columns_tool logic.
    """
    selected_keys = state.get("selected_keys")
    if selected_keys is None:
        selected_keys = []

    result = pick_columns_tool.invoke({
        "query": state["query"],
        "selected_keys": selected_keys,
        "candidate_columns": state["candidate_columns"],
        "llm": state["llm"]
    })
    return result

def check_relevance_node(state: AgentState) -> Dict[str, Any]:
    """
    Checks if the selected keys and columns are relevant to the query.
    Returns a decision flag.
    """
    logger.info("--- Node: Check Relevance ---")
    
    selected_keys = state.get("selected_keys", [])
    selected_columns = state.get("selected_columns", [])
    iteration_count = state.get("iteration_count", 0)
    
    logger.info(f"Iteration: {iteration_count}, Keys: {len(selected_keys)}, Columns: {len(selected_columns)}")
    
    # Always proceed if we have both keys and columns, OR if we've tried twice
    # This prevents infinite loops
    if (selected_keys and selected_columns) or iteration_count >= 2:
        logger.info("Relevance check PASSED. Proceeding to generate response.")
        return {"relevance_passed": True}
    
    logger.info("Relevance check FAILED. Looping back.")
    return {"relevance_passed": False}

def should_loop_back(state: AgentState) -> Literal["planner", "generate_response"]:
    """
    Conditional edge function to decide whether to loop back or proceed.
    """
    relevance_passed = state.get("relevance_passed", False)
    iteration_count = state.get("iteration_count", 0)
    
    logger.info(f"Should loop back? relevance_passed={relevance_passed}, iteration={iteration_count}")
    
    # Force exit after 2 iterations to prevent infinite loops
    if iteration_count >= 2:
        logger.info("Forcing exit: max iterations reached")
        return "generate_response"
    
    if relevance_passed:
        logger.info("Going to generate_response")
        return "generate_response"
    else:
        logger.info("Looping back to planner")
        return "planner"

def generate_response_node(state: AgentState):
    """
    Generates the final LLM response based on selected keys and columns.
    This is where the actual query answering happens.
    """
    logger.info("--- Node: Generate Response ---")
    
    # For now, just return a placeholder
    # In the actual implementation, this would:
    # 1. Filter dataframe based on selected_columns
    # 2. Pass filtered data to LLM
    # 3. Generate final answer
    
    query = state["query"]
    selected_keys = state.get("selected_keys", [])
    selected_columns = state.get("selected_columns", [])
    
    response = f"Generated response for query: '{query}' using keys: {selected_keys} and columns: {selected_columns}"
    
    return {"final_response": response}

# --- Graph Definition ---

def create_graph():
    """
    Creates the LangGraph workflow.
    Flow: Load -> Planner -> Column Picker -> Check Relevance -> [Loop or Generate] -> End
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_mappings", load_mappings_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("column_picker", column_node)
    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # Define edges
    workflow.set_entry_point("load_mappings")
    
    # Flow: Load -> Planner -> Column -> Check
    workflow.add_edge("load_mappings", "planner")
    workflow.add_edge("planner", "column_picker")
    workflow.add_edge("column_picker", "check_relevance")
    
    # Conditional edge from check_relevance
    workflow.add_conditional_edges(
        "check_relevance",
        should_loop_back,
        {
            "planner": "planner",  # Loop back
            "generate_response": "generate_response"  # Proceed
        }
    )
    
    # Final edge
    workflow.add_edge("generate_response", END)
    
    # Add memory
    memory = MemorySaver()
    
    # Compile
    app = workflow.compile(checkpointer=memory)
    return app

# --- Visualization ---

def get_graph_image(app):
    """
    Returns the graph visualization as a PNG image data.
    """
    try:
        return app.get_graph().draw_mermaid_png()
    except Exception as e:
        logger.error(f"Could not draw graph: {e}")
        return None
