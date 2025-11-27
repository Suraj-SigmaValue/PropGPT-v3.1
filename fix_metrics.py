"""Fix the metrics display section"""
with open('c_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the old metrics display
content = content.replace(
    '''        with col3:
            st.metric("Total Tokens", input_tokens + output_tokens)
        with col4:
            st.metric("Model", f"{provider_display}\\n{model_name}")''',
    '''        with col3:
            st.metric("Total Tokens", input_tokens + output_tokens)
        with col4:
            st.metric("LLMs", f"Map: {mapping_provider_display}\\nResp: {response_provider_display}")
        
        # LLM-wise token breakdown
        st.markdown("---")
        st.markdown("**Token Usage by LLM:**")
        tok_col1, tok_col2 = st.columns(2)
        with tok_col1:
            st.markdown(f"""
            <div style='background-color: #1A1F2E; padding: 1rem; border-radius: 6px;'>
                <p style='color: #6BA3F5; margin: 0; font-size: 0.85rem;'>📊 <strong>Mapping LLM</strong> ({mapping_provider_display})</p>
                <p style='color: #E8EAED; margin: 0.5rem 0 0 0; font-size: 1.25rem;'>{input_tokens} tokens</p>
                <p style='color: #B0B5C2; margin: 0; font-size: 0.75rem;'>Column selection & mapping</p>
            </div>
            """, unsafe_allow_html=True)
        with tok_col2:
            st.markdown(f"""
            <div style='background-color: #1A1F2E; padding: 1rem; border-radius: 6px;'>
                <p style='color: #6BA3F5; margin: 0; font-size: 0.85rem;'>🤖 <strong>Response LLM</strong> ({response_provider_display})</p>
                <p style='color: #E8EAED; margin: 0.5rem 0 0 0; font-size: 1.25rem;'>{output_tokens} tokens</p>
                <p style='color: #B0B5C2; margin: 0; font-size: 0.75rem;'>Analysis generation</p>
            </div>
            """, unsafe_allow_html=True)'''
)

with open('c_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed metrics display!")
