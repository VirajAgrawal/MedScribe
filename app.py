"""
Healthcare GenAI Tool — Simplified
Stack: Python + Flask | ChromaDB (Vector DB) | Gemini LLM API
"""

import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__, static_folder="static")

# ── Gemini Setup ──
# Get your free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCrMFf7VmMF-8F1Y60GRUq_rynKidrdKco")
genai.configure(api_key=GEMINI_API_KEY)

# ── ChromaDB Vector Store ──
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="health_docs",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# ── Prompt Templates (Prompt Engineering) ──
PROMPTS = {
    "patient_summary": """
You are a clinical documentation specialist.
Generate a structured PATIENT SUMMARY using SOAP format.

Rules:
- Use formal medical terminology
- Sections: Chief Complaint, History, Findings, Assessment, Plan
- Format sections with ## headers
- Be concise and professional

Similar past cases from database:
{context}

Patient details:
- Name: {name}
- Age: {age} | Gender: {gender}
- Chief Complaint: {complaint}
- Medical History: {history}
- Current Medications: {medications}
- Additional Notes: {notes}
""",

    "medical_report": """
You are a senior medical officer writing a formal MEDICAL REPORT.

Rules:
- Structure: Summary → Clinical Findings → Diagnosis → Treatment Plan → Prognosis
- Use authoritative, evidence-based language
- Format sections with ## headers
- Be thorough and precise

Similar past cases from database:
{context}

Patient details:
- Name: {name}
- Age: {age} | Gender: {gender}
- Chief Complaint: {complaint}
- Medical History: {history}
- Current Medications: {medications}
- Additional Notes: {notes}
"""
}

def get_context(query):
    """Retrieve similar past documents from Vector DB (RAG)."""
    try:
        if collection.count() == 0:
            return "No prior cases in database."
        results = collection.query(query_texts=[query], n_results=min(2, collection.count()))
        docs = results["documents"][0]
        return "\n".join(f"- {d[:250]}..." for d in docs) if docs else "No similar cases found."
    except:
        return "Context unavailable."

def save_to_db(input_text, output_text, doc_type):
    """Store generated document in Vector DB for future retrieval."""
    collection.add(
        documents=[f"INPUT: {input_text} OUTPUT: {output_text}"],
        metadatas=[{"doc_type": doc_type, "date": datetime.now().isoformat()}],
        ids=[str(uuid.uuid4())]
    )

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    doc_type = data.get("doc_type", "patient_summary")

    fields = {
        "name":        data.get("name", "").strip(),
        "age":         data.get("age", "").strip(),
        "gender":      data.get("gender", "").strip(),
        "complaint":   data.get("complaint", "").strip(),
        "history":     data.get("history", "").strip(),
        "medications": data.get("medications", "N/A").strip(),
        "notes":       data.get("notes", "None").strip(),
    }

    if not all([fields["name"], fields["age"], fields["complaint"]]):
        return jsonify({"success": False, "error": "Name, age, and chief complaint are required."}), 400

    query_text = f"{fields['complaint']} {fields['history']}"
    context = get_context(query_text)
    prompt = PROMPTS[doc_type].format(context=context, **fields)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        output = response.text
        save_to_db(query_text, output, doc_type)
        return jsonify({
            "success": True,
            "content": output,
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Running at http://localhost:5000")
    app.run(debug=True, port=5000)
