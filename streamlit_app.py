import streamlit as st
import re
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException

# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Indian Legal RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# 2. INITIALIZE SNOWFLAKE SESSION
# -------------------------------------------------
try:
    session = get_active_session()
except:
    st.error("⚠️ Could not get active Snowflake session. Make sure you are running this inside Streamlit in Snowflake.")
    st.stop()

# -------------------------------------------------
# 3. SIDEBAR UI
# -------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    # "New Chat" Button
    if st.button("🧹 New Chat", use_container_width=True):
        st.rerun()
        
    st.markdown("---")
    
    # Debug Toggle
    show_debug = st.checkbox("Show tool debug", value=False)
    
    st.markdown("---")
    
    # About Section
    st.markdown("### ℹ️ About")
    st.info(
        """
        * **Tool-calling RAG**
        * **Strict grounding**
        * **Citation aware**
        """
    )

# -------------------------------------------------
# 4. HELPER FUNCTIONS
# -------------------------------------------------

def detect_section(question):
    """Extracts section number from the user's question."""
    match = re.search(r"Section\s+(\d+)", question, re.IGNORECASE)
    return match.group(1) if match else None

def detect_act(question):
    """Maps keywords in the question to specific filenames."""
    q = question.lower()
    
    if "immigration" in q or "foreigner" in q:
        return "Immigration_and_Foreigners_Act_2025.json"
    if "nyaya" in q or "bns" in q:
        return "Bharatiya_Nyaya_Sanhita_2023.json"
    if "suraksha" in q or "bnss" in q:
        return "Bharatiya_Nagarik_Suraksha_Sanhita_2023.json"
    if "ipc" in q or "penal code" in q:
        return "The_Indian_Penal_Code_1860.json"
    if "evidence" in q or "bsa" in q:
        return "The_Indian_Evidence_Act_1872.json"
        
    return None

def clean_statutory_text(text):
    """Cleans raw text to make it readable for the AI."""
    if not text: return ""
    text = text.replace(";", "; ").replace(":", ": ")
    text = re.sub(r'(?<=[a-z0-9])\(', ' (', text)
    text = text.replace("-(", " - (")
    return text

def retrieve_sections(question):
    """Retrieves legal text (Exact + Semantic)."""
    section_number = detect_section(question)
    act_name = detect_act(question)
    
    retrieved_rows = []
    
    # --- Strategy A: Exact Section Match (Deterministic) ---
    if section_number:
        base_query = """
            SELECT SECTION_NUMBER, FILENAME, SECTION_TEXT, 1.0 AS SCORE 
            FROM LEGAL_SECTIONS 
            WHERE SECTION_NUMBER = ?
        """
        params = [section_number]
        
        if act_name:
            base_query += " AND FILENAME = ?"
            params.append(act_name)
            
        # Collect ALL chunks (No LIMIT)
        df_det = session.sql(base_query, params=params).collect()
        for row in df_det:
            retrieved_rows.append(row.as_dict())

    # --- Strategy B: Semantic Search (Vector) ---
    if len(retrieved_rows) == 0:
        safe_question = question.replace("'", "''") 
        
        semantic_query = f"""
            WITH query_embedding AS (
                SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                    'snowflake-arctic-embed-m-v1.5', 
                    '{safe_question}'
                ) AS q_vec
            )
            SELECT SECTION_NUMBER, FILENAME, SECTION_TEXT, 
                   VECTOR_COSINE_SIMILARITY(EMBEDDING, q_vec) AS SCORE
            FROM LEGAL_SECTIONS, query_embedding
            WHERE SCORE > 0.75
            AND LENGTH(SECTION_TEXT) > 100
        """
        if act_name:
            semantic_query += f" AND FILENAME = '{act_name}'"
        semantic_query += " ORDER BY SCORE DESC LIMIT 5"
        
        df_sem = session.sql(semantic_query).collect()
        for row in df_sem:
            retrieved_rows.append(row.as_dict())

    return retrieved_rows

def generate_answer(question, context_text):
    """Generates an answer using Snowflake Cortex LLM with anti-repetition logic."""
    
    # 1. Clean the text
    clean_context = clean_statutory_text(context_text)
    
    # 2. Safety Truncation (Max ~3000 tokens)
    if len(clean_context) > 12000:
        clean_context = clean_context[:12000] + "\n... [Context Truncated]"
    
    # 3. Escape quotes
    safe_question = question.replace("'", "''")
    safe_context = clean_context.replace("'", "''")
    
    # 4. PROMPT (Specifically fixed to stop repetition)
    prompt_text = f"""
    You are an expert legal assistant. 
    
    CONTEXT (Statutory Law):
    {safe_context}
    
    USER QUESTION: 
    {safe_question}
    
    INSTRUCTIONS:
    1. **If this is a "Definitions" section (e.g., Section 2):**
       - Do NOT output a raw list of words.
       - Do NOT repeat words like "harbour", "injury", "life".
       - Instead, write a summary paragraph: "Section 2 provides definitions for key terms used in the Act, such as 'Child', 'Document', 'Fraudulently', and 'Public Servant'."
       - You may list 3-4 important examples, but do not list everything.
       
    2. **If this is about Powers/Procedures:**
       - Use bullet points to list them clearly.
       
    3. **General Rules:**
       - Write in complete, descriptive sentences.
       - Cite the Section Number and Act Name.
       - If the answer is not in the context, say: "Insufficient information in provided statutory documents."
    """
    
    query = "SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', ?) AS ANSWER"
    try:
        result = session.sql(query, params=[prompt_text]).collect()
        return result[0]["ANSWER"]
    except SnowparkSQLException as e:
        return f"Error generating answer: {str(e)}"

# -------------------------------------------------
# 5. MAIN INTERFACE
# -------------------------------------------------

st.title("📚 Indian Legal RAG Assistant")
st.markdown("Strictly grounded legal answers with citations")

if prompt := st.chat_input("Ask a legal question (e.g. What is Section 1 of BNSS, 2023?)"):
    
    # Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing statutory texts..."):
            
            # 1. Retrieve Data
            retrieved_data = retrieve_sections(prompt)
            
            # Debug View
            if show_debug:
                with st.status("🛠️ Tool Debug", expanded=False):
                    st.write(f"Detected Section: {detect_section(prompt)}")
                    st.write(f"Detected Act: {detect_act(prompt)}")
                    st.write(f"Retrieved Chunks: {len(retrieved_data)}")

            if not retrieved_data:
                st.warning("No relevant statutory sections found.")
                st.stop()

            # 2. Build Context
            context_text = "\n\n".join([
                f"--- SOURCE: Section {row['SECTION_NUMBER']} ({row['FILENAME']}) ---\n{row['SECTION_TEXT']}"
                for row in retrieved_data
            ])
            
            # 3. Generate Answer
            answer = generate_answer(prompt, context_text)
            
            # 4. Display Result
            st.markdown(answer)
            
            # 5. Display Sources
            with st.expander("📚 View Statutory Sources", expanded=False):
                for row in retrieved_data:
                    st.markdown(f"**Section {row['SECTION_NUMBER']} — {row['FILENAME']}**")
                    if 'SCORE' in row:
                        st.caption(f"Relevance Score: {round(row['SCORE'], 3)}")
                    st.text(clean_statutory_text(row['SECTION_TEXT']))
                    st.markdown("---")
