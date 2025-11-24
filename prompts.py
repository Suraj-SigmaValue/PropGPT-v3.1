"""
Prompt templates for PropGPT real-estate analysis.
Contains prompts for different comparison types:
- Location-wise analysis (with year-wise trends)
- City-wise analysis (with year-wise trends)
- Project-wise analysis (no year-wise trends)
"""

from typing import List


def build_location_prompt(
    question: str,
    items: List[str],
    mapping_keys: List[str],
    selected_columns: List[str],
    context: str,
    category_summary: str,
) -> str:
    """
    Build prompt for location-wise comparison analysis.
    Includes year-wise trends (2020-2024).
    """
    # Create items list for display
    if len(items) == 1:
        items_display = items[0]
    elif len(items) == 2:
        items_display = f"{items[0]} vs {items[1]}"
    else:
        items_display = ", ".join(items[:-1]) + f" and {items[-1]}"

    return f"""
You are PropGPT, an elite real-estate analyst. Answer in clean, well-structured MARKDOWN.

MARKDOWN FORMATTING RULES (CRITICAL):
1. Use ### for main section headings (Executive Summary, Trend Highlights, Deep Insights)
2. Use #### for subsection headings (location names like Baner, Hinjewadi, Ravet, or categories like Flats, Offices)
3. Use - for bullet points (NOT * or ◦)
4. NEVER use bold asterisks (**text**) or italic asterisks (*text*) for headings
5. Always put a blank line between major sections only
6. NO blank lines after subheadings - content should start immediately
7. NO TABLES - use bullet points instead
8. NO separate year bullets - integrate year info into content

REQUEST
- Query: "{question}"
- Comparison Type: Location
- {"Analyze" if len(items) == 1 else "Compare"}: {items_display} (location-wise, 2020–2024)
- Number of Items: {len(items)}
- Categories: {category_summary}

RETRIEVED EVIDENCE (context)
{context}

IMPORTANT - VALUE PRESERVATION:
Use the EXACT values from the RETRIEVED EVIDENCE above. Do NOT convert, reformat, or reinterpret the numbers.
If a value shows "21887359110", use it exactly as 21887359110 in your response.
Do NOT convert to crores, millions, or any other unit unless explicitly shown in the data.

OUTPUT FORMAT (STRICT - follow exactly):

# [Title/Topic Name]

### Executive Summary
[2-3 sentences summarizing key findings]

### {"Key Metrics" if len(items) == 1 else "Trend Highlights"}
[Bullet points with key metrics and trends]

### {"Deep Insights" if len(items) == 1 else "Deep Insights"}

CRITICAL DO's AND DON'Ts:
✓ DO: Use exact values from evidence (e.g., "Total sales (2020): ₹21887359110 → (2024): ₹28234431307")
✗ DON'T: Convert or reformat the numbers (e.g., NO crores conversion)
✓ DO: Integrate years directly into the metric statement
✗ DON'T: Add blank lines after #### headings
✓ DO: Start content immediately after heading
✗ DON'T: Use bare year numbers as bullet points
"""


def build_city_prompt(
    question: str,
    items: List[str],
    mapping_keys: List[str],
    selected_columns: List[str],
    context: str,
    category_summary: str,
) -> str:
    """
    Build prompt for city-wise comparison analysis.
    Includes year-wise trends (2020-2024).
    """
    # Create items list for display
    if len(items) == 1:
        items_display = items[0]
    elif len(items) == 2:
        items_display = f"{items[0]} vs {items[1]}"
    else:
        items_display = ", ".join(items[:-1]) + f" and {items[-1]}"

    return f"""
You are PropGPT, an elite real-estate analyst. Answer in clean, well-structured MARKDOWN.

MARKDOWN FORMATTING RULES (CRITICAL):
1. Use ### for main section headings (Executive Summary, Trend Highlights, Deep Insights)
2. Use #### for subsection headings (city names or categories like Flats, Offices)
3. Use - for bullet points (NOT * or ◦)
4. NEVER use bold asterisks (**text**) or italic asterisks (*text*) for headings
5. Always put a blank line between major sections only
6. NO blank lines after subheadings - content should start immediately
7. NO TABLES - use bullet points instead
8. NO separate year bullets - integrate year info into content

REQUEST
- Query: "{question}"
- Comparison Type: City
- {"Analyze" if len(items) == 1 else "Compare"}: {items_display} (city-wise, 2020–2024)
- Number of Items: {len(items)}
- Categories: {category_summary}

RETRIEVED EVIDENCE (context)
{context}

IMPORTANT - VALUE PRESERVATION:
Use the EXACT values from the RETRIEVED EVIDENCE above. Do NOT convert, reformat, or reinterpret the numbers.
If a value shows "21887359110", use it exactly as 21887359110 in your response.
Do NOT convert to crores, millions, or any other unit unless explicitly shown in the data.

OUTPUT FORMAT (STRICT - follow exactly):

# [Title/Topic Name]

### Executive Summary
[2-3 sentences summarizing key findings]

### {"Key Metrics" if len(items) == 1 else "Trend Highlights"}
[Bullet points with key metrics and trends]

### {"Deep Insights" if len(items) == 1 else "Deep Insights"}
{"" if len(items) == 1 else f"""
For each city, include:

#### [City Name]
#### Flats:
- Average Price (2020): ₹X per sq. ft. → (2024): ₹Y per sq. ft. (Growth: Z%)
- [Key insight about trend]
- [Another metric or observation]

#### Offices:
- Average Price (2020): ₹X per sq. ft. → (2024): ₹Y per sq. ft. (Growth: Z%)
- [Key insight about trend]
"""}

CRITICAL DO's AND DON'Ts:
✓ DO: Use exact values from evidence (e.g., "Total sales (2020): ₹21887359110 → (2024): ₹28234431307")
✗ DON'T: Convert or reformat the numbers (e.g., NO crores conversion)
✓ DO: Integrate years directly into the metric statement
✗ DON'T: Add blank lines after #### headings
✓ DO: Start content immediately after heading
✗ DON'T: Use bare year numbers as bullet points
"""


def build_project_prompt(
    question: str,
    items: List[str],
    mapping_keys: List[str],
    selected_columns: List[str],
    context: str,
    category_summary: str,
) -> str:
    """
    Build prompt for project-wise comparison analysis.
    NO year-wise trends (no 2020-2024 mentioned).
    """
    # Create items list for display
    if len(items) == 1:
        items_display = items[0]
    elif len(items) == 2:
        items_display = f"{items[0]} vs {items[1]}"
    else:
        items_display = ", ".join(items[:-1]) + f" and {items[-1]}"

    return f"""
You are PropGPT, an elite real-estate analyst. Answer in clean, well-structured MARKDOWN.

MARKDOWN FORMATTING RULES (CRITICAL):
1. Use ### for main section headings (Executive Summary, Key Metrics, Deep Insights)
2. Use #### for subsection headings (project names, categories like Flats, Offices)
3. Use - for bullet points (NOT * or ◦)
4. NEVER use bold asterisks (**text**) or italic asterisks (*text*) for headings
5. Always put a blank line between major sections only
6. NO blank lines after subheadings - content should start immediately
7. NO TABLES - use bullet points instead
8. NO separate year bullets - integrate any time info into content if present

REQUEST
- Query: "{question}"
- Comparison Type: Project
- {"Analyze" if len(items) == 1 else "Compare"}: {items_display} (project-wise)
- Number of Items: {len(items)}
- Categories: {category_summary}

RETRIEVED EVIDENCE (context)
{context}

IMPORTANT - VALUE PRESERVATION:
Use the EXACT values from the RETRIEVED EVIDENCE above. Do NOT convert, reformat, or reinterpret the numbers.
If a value shows "21887359110", use it exactly as 21887359110 in your response.
Do NOT convert to crores, millions, or any other unit unless explicitly shown in the data.

OUTPUT FORMAT (STRICT - follow exactly):

# [Title/Topic Name]

### Executive Summary
[2-3 sentences summarizing key findings]

### {"Key Metrics" if len(items) == 1 else "Key Metrics"}
[Bullet points with key metrics and trends]

### {"Deep Insights" if len(items) == 1 else "Deep Insights"}
{"" if len(items) == 1 else f"""
For each project, include:

#### [Project Name]
#### Flats:
- [Key metric or insight about flats]
- [Another metric or observation]

#### Offices:
- [Key metric or insight about offices]
- [Another metric or observation]
"""}

CRITICAL DO's AND DON'Ts:
✓ DO: Use exact values from evidence (e.g., "Total sales: ₹21887359110")
✗ DON'T: Convert or reformat the numbers (e.g., NO crores conversion)
✓ DO: Integrate any time info directly into the metric statement if present
✗ DON'T: Add blank lines after #### headings
✓ DO: Start content immediately after heading
✗ DON'T: Use bare year numbers as bullet points
"""
