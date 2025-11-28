import os

def fix_file():
    file_path = 'c_app.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    
    # The code to insert
    chat_logic = [
        '# Initialize chat history\n',
        'if "messages" not in st.session_state:\n',
        '    st.session_state.messages = []\n',
        '\n',
        '# Display chat messages from history\n',
        'for message in st.session_state.messages:\n',
        '    with st.chat_message(message["role"]):\n',
        '        st.markdown(message["content"])\n',
        '\n',
        '# Accept user input\n',
        'if query := st.chat_input("Ask a question about the selected items..."): \n',
        '    st.session_state.messages.append({"role": "user", "content": query})\n',
        '    \n',
        '    with st.chat_message("user"):\n',
        '        st.markdown(query)\n',
        '\n',
        '    with st.chat_message("assistant"):\n',
        '        response_text = None\n',
        '        # Relevance check first\n',
        '        try:\n',
        '            temp_llm = get_llm(selected_response_llm_provider)\n',
        '            if not is_query_relevant(query, temp_llm):\n',
        '                irrelevant_msg = """I appreciate your question, but I\'m specifically designed to help with **real estate analysis** topics such as:\n',
        '\n',
        '- Property market trends and pricing\n',
        '- Sales and demand analysis  \n',
        '- Location and project comparisons\n',
        '- Supply and inventory metrics\n',
        '\n',
        'Could you please ask a question related to real estate or property analysis? \U0001F3E1"""\n',
        '                \n',
        '                st.warning(irrelevant_msg)\n',
        '                st.session_state.messages.append({"role": "assistant", "content": irrelevant_msg})\n',
        '                st.stop()\n',
        '        except Exception as e:\n',
        '            logger.warning(f"Relevance check error: {e}")\n',
        '\n',
        '            # Input validation\n',
        '            if not selected_items:\n',
        '                st.error(f"Please select at least one {comparison_type} to analyze.")\n',
        '                st.stop()\n',
        '    \n',
        '            if len(selected_items) > 5:\n',
        '                st.error(f"Maximum 5 {comparison_type}s can be analyzed at once.")\n',
        '                st.stop()\n',
        '    \n',
        '            # Check for duplicates (should be prevented by multiselect but verify)\n',
        '            if len(set(i.lower() for i in selected_items)) != len(selected_items):\n',
        '                st.markdown(f"""\n',
        '                    <div style=\'background-color: #fce4e4; color: #cc0033; padding: 1rem; border-radius: 8px; margin: 1rem 0;\'>\n',
        '                        <strong>❌ Error:</strong> Please select different {comparison_type}s for comparison.""")\n',
        '\n',
        '            if not categories:\n',
        '                st.error("Please select at least one category to analyze.")\n',
        '                st.stop()\n',
        '\n',
        '            # Query guaranteed from chat_input\n',
        '\n'
    ]

    start_marker = '# ---------------- CHAT INTERFACE ----------------'
    end_marker = 'logger.info("Query: \'%s\'", query)'
    
    in_replacement_zone = False
    replaced = False
    
    for line in lines:
        if start_marker in line:
            new_lines.append(line)
            new_lines.extend(chat_logic)
            in_replacement_zone = True
            replaced = True
            continue
            
        if end_marker in line:
            in_replacement_zone = False
            new_lines.append(line)
            continue
            
        if not in_replacement_zone:
            new_lines.append(line)
    
    if replaced:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("Successfully patched c_app.py")
    else:
        print("Could not find target location to patch.")

if __name__ == "__main__":
    fix_file()
