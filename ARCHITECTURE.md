# PropGPT Code Organization Summary

## File Structure

The application is now organized into modular components:

### 1. **c_app.py** (Main Application)
- Streamlit UI and user interface
- Data loading and processing
- LLM initialization and configuration
- Main analysis flow and orchestration
- Imports agents and prompts from separate modules

### 2. **prompts.py** (Prompt Templates)
Contains three specialized prompt builders:
- **`build_location_prompt()`** - For location-wise analysis (includes year-wise trends 2020-2024)
- **`build_city_prompt()`** - For city-wise analysis (includes year-wise trends 2020-2024)
- **`build_project_prompt()`** - For project-wise analysis (NO year-wise trends)

Each prompt is a complete template with formatting rules, request context, and output format specifications.

### 3. **agents.py** (Query Intelligence Agents)
Contains two LLM-powered agents for intelligent query processing:

#### Planner Agent
- **Function**: `planner_identify_mapping_keys(llm, query, candidate_keys)`
- **Purpose**: Analyzes user query and selects the most relevant mapping keys
- **Returns**: List of selected mapping keys (max 6)
- **Fallback**: Keyword-based heuristic matching if LLM fails

#### Column Agent
- **Function**: `agent_pick_relevant_columns(llm, query, selected_keys, candidate_columns)`
- **Purpose**: Selects relevant columns from candidates based on query and selected mapping keys
- **Returns**: List of selected column names (typically 5-20)
- **Fallback**: Keyword-based heuristic matching if LLM fails

---

## Key Benefits

✅ **Separation of Concerns**: Each module has a single responsibility
- Prompts are managed independently for easy updates
- Agents are isolated for testing and reuse
- Main app focuses on orchestration and UI

✅ **Maintainability**: Changes to prompts or agents don't affect other components

✅ **Reusability**: Agents and prompts can be reused in other projects

✅ **Scalability**: Easy to add new prompt templates or agents

---

## Usage Flow

```
1. User provides query and selects parameters in Streamlit UI (c_app.py)
2. Planner Agent (agents.py) selects relevant mapping keys
3. Column Agent (agents.py) selects relevant columns
4. build_prompt() routes to correct prompt builder based on comparison type (prompts.py)
5. Analysis results are generated and displayed
```

---

## Import Statements

In `c_app.py`:
```python
from prompts import (
    build_location_prompt,
    build_city_prompt,
    build_project_prompt,
)

from agents import (
    planner_identify_mapping_keys,
    agent_pick_relevant_columns,
)
```

---

## All Values Preserved

✅ No changes to knowledge layer or existing logic
✅ All prompts maintain original content and formatting
✅ Agent behavior and fallback logic unchanged
✅ Syntax validated - all files compile successfully
