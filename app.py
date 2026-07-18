import os
import re
import numpy as np
import snowflake.connector
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load model
print("Loading embedding model...")
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
print(f"Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")

# Connect to Snowflake
print("Connecting to Snowflake...")
conn = snowflake.connector.connect(
    account="pfb17012",
    user="ABHIC",
    password="Qwertyuiop@123",
    database="LAW_FILES",
    schema="DATA",
    warehouse="COMPUTE_WH"
)
print("Connected!\n")

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

def get_answers(question, top_k=5):
    cursor = conn.cursor()
    section_number = detect_section(question)
    act_name = detect_act(question)
    results = []

    # Strategy 1 : Exact Section Match
    if section_number:
        sql = """
        SELECT SECTION_NUMBER, FILENAME, SECTION_TEXT, 1.0 AS SCORE
        FROM LEGAL_SECTIONS WHERE SECTION_NUMBER=%s
        """
        params = [section_number]
        if act_name:
            sql += " AND FILENAME=%s"
            params.append(act_name)
        sql += """
        ORDER BY CASE WHEN SECTION_TEXT LIKE CONCAT('%%', %s, '. - %%') THEN 1 ELSE 0 END, ID LIMIT 1
        """
        params.append(section_number)
        cursor.execute(sql, params)
        results = cursor.fetchall()

    # Strategy 2 : Semantic Search
    if not results:
        query_prefix = "Represent this sentence for searching relevant passages: "
        full_query = query_prefix + question.strip()
        q_embedding = model.encode(full_query, normalize_embeddings=True)
        vec_str = "[" + ",".join(f"{v:.8f}" for v in q_embedding.tolist()) + "]"

        sql = f"""
        SELECT SECTION_NUMBER, FILENAME, SECTION_TEXT,
               VECTOR_COSINE_SIMILARITY(EMBEDDING, {vec_str}::VECTOR(FLOAT,768)) AS SCORE
        FROM LEGAL_SECTIONS
        WHERE EMBEDDING IS NOT NULL AND LENGTH(SECTION_TEXT) > 100
        """
        if act_name:
            sql += f" AND FILENAME='{act_name}'"
        sql += f" ORDER BY SCORE DESC LIMIT {top_k}"
        cursor.execute(sql)
        results = cursor.fetchall()    
         
    # Strategy 3 : Keyword Fallback
    if not results or (results and results[0][3] < 0.65):
        keywords = re.findall(r"[a-zA-Z]+", question.lower())
        stop_words = {"what", "does", "the", "is", "of", "state", "regarding", "about", "explain", "under", "section", "act", "code", "bill", "sanhita", "bharatiya", "nyaya", "nagarik", "suraksha", "indian", "penal", "evidence", "immigration", "foreigners", "provide", "how", "which", "tell", "describe"}
        keywords = [w for w in keywords if w not in stop_words and len(w) > 2][:3]

        if keywords:
            conditions = " AND ".join([f"SECTION_TEXT ILIKE '%{kw}%'" for kw in keywords])
            sql = f"""
            SELECT SECTION_NUMBER, FILENAME, SECTION_TEXT, 0.8 AS SCORE
            FROM LEGAL_SECTIONS
            WHERE {conditions} AND LENGTH(SECTION_TEXT) > 100
            """
            if act_name: sql += f" AND FILENAME='{act_name}'"
            sql += " ORDER BY LENGTH(SECTION_TEXT) DESC LIMIT 5"
            cursor.execute(sql)
            kw_results = cursor.fetchall()
            if kw_results: results = kw_results
            
    cursor.close()

    # Format output for the web interface
    formatted_results = []
    for section_num, filename, text, score in results:
        fname = filename.replace(".json", "").replace("_", " ")
        formatted_results.append({
            "section": section_num,
            "act": fname,
            "text": text,
            "score": round(float(score), 4)
        })
        
    return formatted_results

# Web Routes
@app.route('/')
def home():
    # Renders the HTML page
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # Handles the search request from the web page
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400
        
    results = get_answers(question)
    return jsonify(results)

if __name__ == '__main__':
    print("\nServer starting! Open http://127.0.0.1:5000 in your web browser.")
    app.run(debug=True, use_reloader=False)
