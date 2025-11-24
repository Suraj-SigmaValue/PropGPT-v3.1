# PropGPT Sigmavalue

**PropGPT Sigmavalue** is an AI-powered real estate analysis tool designed to provide intelligent insights into property data. It leverages Large Language Models (LLMs) to interpret user queries and generate detailed reports on locations, cities, and specific projects.

## Features

-   **Intelligent Query Processing**: Uses LLM-powered agents to understand natural language queries and select relevant data mapping keys and columns.
-   **Multi-Level Analysis**:
    -   **Location-wise**: Analyze trends and data for specific locations (includes year-wise trends 2020-2024).
    -   **City-wise**: Aggregate data at the city level.
    -   **Project-wise**: Detailed analysis of specific real estate projects.
-   **Interactive UI**: Built with Streamlit for a responsive and user-friendly experience.
-   **Flexible LLM Support**: Supports OpenAI (default), Google Gemini, and NVIDIA NIM.
-   **Hybrid Retrieval**: Combines FAISS vector search and BM25 for accurate document retrieval.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  Create a `.env` file in the root directory.
2.  Add your API keys:

    ```env
    # Required for default configuration
    OPENAI_API_KEY=sk-...

    # Optional: For Google Gemini
    GOOGLE_API_KEY=...

    # Optional: For NVIDIA NIM
    NVIDIA_API_KEY=nvapi-...
    ```

3.  (Optional) Set the LLM provider explicitly:

    ```env
    USE_LLM=openai  # Options: openai, gemini
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run c_app.py
```

## File Structure

-   **`c_app.py`**: Main application file containing the Streamlit UI and orchestration logic.
-   **`agents.py`**: Contains the Planner and Column selection agents.
-   **`prompts.py`**: Stores prompt templates for different analysis types.
-   **`config.py`**: Configuration settings and constants.
-   **`mapping.py`**: Data mapping definitions.
-   **`ARCHITECTURE.md`**: Detailed architectural overview.

## Data Sources

The application loads data from `Pune_Grand_Summary.xlsx` and caches it as `Pune_Grand_Summary.pkl` for performance.
