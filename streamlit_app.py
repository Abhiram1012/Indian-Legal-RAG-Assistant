import streamlit as st
import re
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
# 2. INITIALIZE SNOWFLAKE SESSION  (UPDATED)
# -------------------------------------------------
try:
    cnx = st.connection("snowflake")
    session = cnx.session()
except Exception as e:
    st.error(f"⚠️ Could not connect to Snowflake: {e}")
    st.stop()

# -------------------------------------------------
# 3. SIDEBAR UI
# -------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    if st.button("🧹 New Chat", use_container_width=True):
        st.rerun()
        
    st.markdown("---")
    
    show_debug = st.checkbox("Show tool debug", value=False)
    
    st.markdown("---")
    
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
    match = re.search(r"Section\s+(\d+)", question, re.IGNORECASE)
    return match.group(1) if match else None

def detect_act(question):
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
    if not text:
        return ""
    text = text.replace(";", "; ").replace(":", ": ")
    text = re.sub(r'(?<=[a-z0-9])\(', ' (', text)
    text = text.replace("-(", " - (")
    return text

def retrieve_sections(question):
    section_number = detect_section(question)
    act_name = detect_act(question)
    
    retrieved_rows = []
    
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
            
        df_det = session.sql(base_query, params=params).collect()
        for row in df_det:
            retrieved_rows.append(row.as_dict())

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
    
    clean_context = clean_statutory_text(context_text)
    
    if len(clean_context) > 12000:
        clean_context = clean_context[:12000] + "\n... [Context Truncated]"
    
    safe_question = question.replace("'", "''")
    safe_context = clean_context.replace("'", "''")
    
    prompt_text = f"""
    You are an expert legal assistant. 
    
    CONTEXT (Statutory Law):
    {safe_context}
    
    USER QUESTION: 
    {safe_question}
    
    INSTRUCTIONS:
    1. If this is a "Definitions" section:
       - Do NOT output a raw list.
       - Write a summary paragraph.
       
    2. If this is about Powers/Procedures:
       - Use bullet points.
       
    3. General Rules:
       - Write complete sentences.
       - Cite Section Number and Act Name.
       - If not found, say:
         "Insufficient information in provided statutory documents."
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
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing statutory texts..."):
            
            retrieved_data = retrieve_sections(prompt)
            
            if show_debug:
                with st.status("🛠️ Tool Debug", expanded=False):
                    st.write(f"Detected Section: {detect_section(prompt)}")
                    st.write(f"Detected Act: {detect_act(prompt)}")
                    st.write(f"Retrieved Chunks: {len(retrieved_data)}")

            if not retrieved_data:
                st.warning("No relevant statutory sections found.")
                st.stop()

            context_text = "\n\n".join([
                f"--- SOURCE: Section {row['SECTION_NUMBER']} ({row['FILENAME']}) ---\n{row['SECTION_TEXT']}"
                for row in retrieved_data
            ])
            
            answer = generate_answer(prompt, context_text)
            
            st.markdown(answer)
            
            with st.expander("📚 View Statutary Sources", expanded=False):
                for row in retrieved_data:
                    st.markdown(f"**Section {row['SECTION_NUMBER']} — {row['FILENAME']}**")
                    if 'SCORE' in row:
                        st.caption(f"Relevance Score: {round(row['SCORE'], 3)}")
                    st.text(clean_statutory_text(row['SECTION_TEXT']))
                    st.markdown("---")
