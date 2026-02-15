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
# 2. INITIALIZE SNOWFLAKE SESSION (UPDATED FOR STREAMLIT UI)
# -------------------------------------------------
try:
    cnx = st.connection("snowflake")
    session = cnx.session()
except Exception as e:
    st.error(f"⚠️ Could not connect to Snowflake: {e}")
    st.stop()

# -------------------------------------------------
# 3. INITIALIZE CHAT HISTORY
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# 4. SIDEBAR UI
# -------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    show_debug = st.checkbox("Show tool debug", value=False)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("**Strict grounding & Citation aware**")

# -------------------------------------------------
# 5. HELPER FUNCTIONS (UNCHANGED)
# -------------------------------------------------

def detect_section(question):
    match = re.search(r"Section\s+(\d+)", question, re.IGNORECASE)
    return match.group(1) if match else None

def detect_act(question):
    q = question.lower()
    if "immigration" in q or "foreigner" in q: return "Immigration_and_Foreigners_Act_2025.json"
    if "nyaya" in q or "bns" in q: return "Bharatiya_Nyaya_Sanhita_2023.json"
    if "suraksha" in q or "bnss" in q: return "Bharatiya_Nagarik_Suraksha_Sanhita_2023.json"
    if "ipc" in q or "penal code" in q: return "The_Indian_Penal_Code_1860.json"
    if "evidence" in q or "bsa" in q: return "The_Indian_Evidence_Act_1872.json"
    return None

def clean_statutory_text(text):
    if not text: return ""
    text = text.replace(";", "; ").replace(":", ": ")
    text = re.sub(r'(?<=[a-z0-9])\(', ' (', text)
    text = text.replace("-(", " - (")
    return text

def retrieve_sections(question):
    section_number = detect_section(question)
    act_name = detect_act(question)
    retrieved_rows = []
    
    # Exact Match
    if section_number:
        base_query = "SELECT SECTION_NUMBER, FILENAME, SECTION_TEXT, 1.0 AS SCORE FROM LEGAL_SECTIONS WHERE SECTION_NUMBER = ?"
        params = [section_number]
        if act_name:
            base_query += " AND FILENAME = ?"
            params.append(act_name)
        df_det = session.sql(base_query, params=params).collect()
        for row in df_det:
            retrieved_rows.append(row.as_dict())

    # Semantic Search
    if not retrieved_rows:
        safe_q = question.replace("'", "''")
        q = f"""
            WITH query_embedding AS (
                SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m-v1.5', '{safe_q}') AS q_vec
            )
            SELECT SECTION_NUMBER, FILENAME, SECTION_TEXT,
                   VECTOR_COSINE_SIMILARITY(EMBEDDING, q_vec) AS SCORE
            FROM LEGAL_SECTIONS, query_embedding
            WHERE SCORE > 0.75 AND LENGTH(SECTION_TEXT) > 100
        """
        if act_name:
            q += f" AND FILENAME = '{act_name}'"
        q += " ORDER BY SCORE DESC LIMIT 5"
        
        df_sem = session.sql(q).collect()
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
    You are a strict legal assistant. 
    
    CONTEXT (Statutory Law):
    {safe_context}
    
    USER QUESTION: 
    {safe_question}
    
    INSTRUCTIONS:
    1. Output as Structured Bulleted List.
    2. Keep legal language precise.
    3. Append citation at end of every bullet.
    4. If not found say:
       "Insufficient information in provided statutory documents."
    """
    
    query = "SELECT SNOWFLAKE.CORTEX.COMPLETE('snowflake-arctic', ?) AS ANSWER"
    try:
        result = session.sql(query, params=[prompt_text]).collect()
        return result[0]["ANSWER"]
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# -------------------------------------------------
# 6. DISPLAY CHAT HISTORY
# -------------------------------------------------
st.title("📚 Indian Legal RAG Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Statutory Sources", expanded=False):
                for row in message["sources"]:
                    st.markdown(f"**Section {row['SECTION_NUMBER']} — {row['FILENAME']}**")
                    st.text(clean_statutory_text(row['SECTION_TEXT']))
                    st.divider()

# -------------------------------------------------
# 7. MAIN CHAT INPUT
# -------------------------------------------------
if prompt := st.chat_input("Ask a legal question..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing statutes..."):
            
            retrieved_data = retrieve_sections(prompt)
            
            if not retrieved_data:
                answer = "No relevant statutory sections found in the database."
                sources = []
            else:
                context_text = "\n\n".join([
                    f"Source: {r['FILENAME']}\n{r['SECTION_TEXT']}"
                    for r in retrieved_data
                ])
                answer = generate_answer(prompt, context_text)
                sources = retrieved_data 

            st.markdown(answer)
            
            if sources:
                with st.expander("📚 View Statutory Sources", expanded=False):
                    for row in sources:
                        st.markdown(f"**Section {row['SECTION_NUMBER']} — {row['FILENAME']}**")
                        st.text(clean_statutory_text(row['SECTION_TEXT']))
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
