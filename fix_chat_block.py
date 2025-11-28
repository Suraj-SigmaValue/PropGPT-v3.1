import os

def fix_file():
    file_path = 'c_app.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    found_target = False
    
    # The code to insert
    chat_logic = [
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
        '            # Input validation\n'
    ]

    for i, line in enumerate(lines):
        new_lines.append(line)
        if '# Display chat messages from history' in line:
            # Check if the next line is the broken one (indented if)
            if i + 1 < len(lines) and 'if not selected_items:' in lines[i+1]:
                found_target = True
                new_lines.extend(chat_logic)
    
    if found_target:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("Successfully patched c_app.py")
    else:
        print("Could not find target location to patch.")

if __name__ == "__main__":
    fix_file()
