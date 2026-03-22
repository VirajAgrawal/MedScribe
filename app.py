"""
Healthcare GenAI Tool
Stack: Python + Flask | ChromaDB (Vector DB with TF-IDF) | Gemini LLM API

HOW RAG WORKS HERE:
- Every generated document is stored in ChromaDB with a TF-IDF vector embedding
- On each new request, ChromaDB finds the most similar past case using cosine similarity
- That past case is injected into the prompt as context before calling Gemini
- This means the tool gets smarter with every document generated
"""

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

# ── Gemini Setup ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ── ChromaDB with simple embedding function ──
# We use a custom embedding function based on a hash-of-words approach
# that works without any extra ML dependencies (onnxruntime etc.)
class SimpleEmbedder:
    """
    Lightweight deterministic embedder using character n-gram hashing.
    Converts text → a fixed 64-dim float vector for similarity search.
    No external ML packages needed beyond chromadb itself.
    """
    def __call__(self, input):
        results = []
        for text in input:
            vec = [0.0] * 64
            text = text.lower()
            for i in range(len(text) - 2):
                trigram = text[i:i+3]
                h = hash(trigram) % 64
                vec[h] += 1.0
            # Normalize
            total = sum(vec) or 1.0
            vec = [v / total for v in vec]
            results.append(vec)
        return results

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="health_docs",
    embedding_function=SimpleEmbedder(),
    metadata={"hnsw:space": "cosine"}
)

# ── Prompt Templates ──
PROMPTS = {
    "patient_summary": """You are a clinical documentation specialist.
Generate a structured PATIENT SUMMARY using SOAP format.

Rules:
- Use formal medical terminology
- Sections: Chief Complaint, History, Findings, Assessment, Plan
- Format sections with ## headers
- Be concise and professional

--- SIMILAR PAST CASE FROM DATABASE (for reference) ---
{context}
---

Patient details:
- Name: {name}
- Age: {age} | Gender: {gender}
- Chief Complaint: {complaint}
- Medical History: {history}
- Current Medications: {medications}
- Additional Notes: {notes}
""",

    "medical_report": """You are a senior medical officer writing a formal MEDICAL REPORT.

Rules:
- Structure: Summary → Clinical Findings → Diagnosis → Treatment Plan → Prognosis
- Use authoritative, evidence-based language
- Format sections with ## headers

--- SIMILAR PAST CASE FROM DATABASE (for reference) ---
{context}
---

Patient details:
- Name: {name}
- Age: {age} | Gender: {gender}
- Chief Complaint: {complaint}
- Medical History: {history}
- Current Medications: {medications}
- Additional Notes: {notes}
"""
}

# ── Vector DB: Retrieve ──
def get_context(query_text):
    """
    RAG Step 1: Query ChromaDB for the most similar past case.
    Returns a dict with the context string and metadata about what was found.
    """
    count = collection.count()
    if count == 0:
        return {
            "text": "No prior cases in database yet. This is the first entry.",
            "found": False,
            "total_in_db": 0
        }

    results = collection.query(
        query_texts=[query_text],
        n_results=min(2, count)
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return {
            "text": "No similar cases found.",
            "found": False,
            "total_in_db": count
        }

    # Build a readable context snippet
    context_lines = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        snippet = doc[:300].replace("INPUT: ", "").replace("OUTPUT: ", " → ")
        context_lines.append(f"Case {i+1} [{meta.get('doc_type','unknown')} | {meta.get('date','')[:10]}]: {snippet}...")

    return {
        "text": "\n".join(context_lines),
        "found": True,
        "total_in_db": count,
        "matched": len(docs)
    }

# ── Vector DB: Store ──
def save_to_db(query_text, output_text, doc_type, patient_name):
    """
    RAG Step 2: Store this case in ChromaDB after generation.
    The embedding is computed automatically by SimpleEmbedder.
    """
    collection.add(
        documents=[f"INPUT: {query_text} OUTPUT: {output_text[:500]}"],
        metadatas=[{
            "doc_type": doc_type,
            "patient": patient_name,
            "date": datetime.now().isoformat()
        }],
        ids=[str(uuid.uuid4())]
    )

# ── Routes ──
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
        "medications": data.get("medications", "N/A").strip() or "N/A",
        "notes":       data.get("notes", "None").strip() or "None",
    }

    if not all([fields["name"], fields["age"], fields["complaint"]]):
        return jsonify({"success": False, "error": "Name, age, and chief complaint are required."}), 400

    # Step 1: Query Vector DB for similar past case
    query_text = f"{fields['complaint']} {fields['history']} {fields['notes']}"
    rag = get_context(query_text)

    # Step 2: Build engineered prompt with context injected
    prompt = PROMPTS[doc_type].format(context=rag["text"], **fields)

    try:
        # Step 3: Call Gemini
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        output = response.text

        # Step 4: Store result back in Vector DB
        save_to_db(query_text, output, doc_type, fields["name"])

        return jsonify({
            "success": True,
            "content": output,
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
            # RAG info sent to frontend for display
            "rag_found": rag["found"],
            "rag_total": rag["total_in_db"] + 1,  # +1 because we just added
            "rag_context": rag["text"] if rag["found"] else None
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Running at http://localhost:5000")
    app.run(debug=True, port=5000)
