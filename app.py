"""
Healthcare GenAI Content Generation Tool
=========================================
Stack: Python + Flask | ChromaDB (Vector DB) | Gemini LLM API
Purpose: Generate professional healthcare documents using Prompt Engineering
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

# ─────────────────────────────────────────────
# 1. APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder="static")

# ⚠️ Replace with your Gemini API Key (free tier at https://aistudio.google.com)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCaZypmh03Oz-lBOtFRKFPIqScWBvSogjQ")
genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────
# 2. VECTOR DB SETUP (ChromaDB)
# ─────────────────────────────────────────────
# ChromaDB stores past generated summaries as "memory"
# so the tool can retrieve similar past cases as context (RAG pattern)
chroma_client = chromadb.Client()  # In-memory DB (use chromadb.PersistentClient("./db") to persist)

# Use a simple sentence-transformer embedding (or default)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="patient_summaries",
    embedding_function=embedding_fn
)

# ─────────────────────────────────────────────
# 3. PROMPT ENGINEERING TEMPLATES
# ─────────────────────────────────────────────
# This is the core of the Prompt Engineering approach.
# Each document type has a carefully crafted system prompt
# that enforces tone, formatting, and medical terminology.

PROMPT_TEMPLATES = {
    "patient_summary": """
You are a board-certified clinical documentation specialist.
Your task is to generate a structured PATIENT SUMMARY for healthcare professionals.

STRICT RULES:
- Use formal clinical language and standard medical terminology
- Follow SOAP note conventions where applicable (Subjective, Objective, Assessment, Plan)
- Be concise but comprehensive — no filler phrases
- Always include: Chief Complaint, Key History, Findings, Assessment, Recommendations
- Format output with clear section headers using markdown (##)
- Tone: Professional, third-person, objective

CONTEXT FROM SIMILAR PAST CASES (from Vector DB):
{vector_context}

NOW GENERATE A PATIENT SUMMARY FOR:
{user_input}
""",

    "discharge_note": """
You are a hospital discharge coordinator and clinical writer.
Generate a formal DISCHARGE NOTE for healthcare records.

STRICT RULES:
- Include: Admission Reason, Hospital Course, Discharge Condition, Medications, Follow-up Instructions
- Use passive clinical voice ("Patient was admitted...", "Treatment included...")
- Avoid jargon not standard in hospital documentation
- Format with markdown section headers (##)
- Tone: Formal, clear, medico-legal appropriate

CONTEXT FROM SIMILAR PAST CASES (from Vector DB):
{vector_context}

GENERATE DISCHARGE NOTE FOR:
{user_input}
""",

    "clinical_referral": """
You are a specialist physician writing a CLINICAL REFERRAL LETTER.

STRICT RULES:
- Address to: Specialist (unnamed unless specified)
- Include: Patient background, Reason for referral, Relevant history, Urgency level
- Use courteous but direct professional tone
- Format with markdown section headers (##)
- End with a polite closing and "Referring Physician" signature block

CONTEXT FROM SIMILAR PAST CASES (from Vector DB):
{vector_context}

GENERATE CLINICAL REFERRAL FOR:
{user_input}
""",

    "medical_report": """
You are a senior medical officer writing a FORMAL MEDICAL REPORT.

STRICT RULES:
- Structure: Executive Summary → Clinical Findings → Diagnosis → Treatment Plan → Prognosis
- Use ICD-10 style diagnostic language where applicable
- Avoid ambiguity — be definitive in assessments
- Format with markdown section headers (##)
- Tone: Authoritative, evidence-based, professional

CONTEXT FROM SIMILAR PAST CASES (from Vector DB):
{vector_context}

GENERATE MEDICAL REPORT FOR:
{user_input}
"""
}

# ─────────────────────────────────────────────
# 4. CORE FUNCTIONS
# ─────────────────────────────────────────────

def retrieve_similar_context(query_text: str, n_results: int = 2) -> str:
    """
    Vector DB Retrieval (RAG Step):
    Query ChromaDB for past summaries similar to the current input.
    Returns formatted context string to inject into the prompt.
    """
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, collection.count())
        )
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            context_parts = [f"- {doc[:300]}..." for doc in docs]
            return "\n".join(context_parts)
    except Exception:
        pass
    return "No similar past cases found in the database yet."


def store_in_vector_db(input_text: str, output_text: str, doc_type: str):
    """
    Vector DB Storage:
    Save the generated summary into ChromaDB for future retrieval.
    This builds up the tool's 'memory' over time.
    """
    doc_id = str(uuid.uuid4())
    metadata = {
        "doc_type": doc_type,
        "timestamp": datetime.now().isoformat(),
        "input_preview": input_text[:100]
    }
    # Store a combined representation for embedding
    combined_text = f"INPUT: {input_text}\nOUTPUT: {output_text}"
    collection.add(
        documents=[combined_text],
        metadatas=[metadata],
        ids=[doc_id]
    )
    return doc_id


def generate_healthcare_content(user_input: str, doc_type: str) -> dict:
    """
    Main Generation Pipeline:
    1. Retrieve similar past cases from Vector DB (RAG)
    2. Build engineered prompt with context
    3. Call Gemini LLM API
    4. Store result back in Vector DB
    5. Return response
    """
    # Step 1: Retrieve context from Vector DB
    vector_context = retrieve_similar_context(user_input)

    # Step 2: Build the engineered prompt
    template = PROMPT_TEMPLATES.get(doc_type, PROMPT_TEMPLATES["patient_summary"])
    final_prompt = template.format(
        vector_context=vector_context,
        user_input=user_input
    )

    # Step 3: Call Gemini API (free tier model)
    model = genai.GenerativeModel("gemini-2.5-flash")  # Free tier model
    response = model.generate_content(final_prompt)
    generated_text = response.text

    # Step 4: Store in Vector DB for future context
    doc_id = store_in_vector_db(user_input, generated_text, doc_type)

    return {
        "success": True,
        "doc_type": doc_type,
        "generated_content": generated_text,
        "vector_db_id": doc_id,
        "similar_context_used": vector_context,
        "timestamp": datetime.now().strftime("%B %d, %Y – %H:%M")
    }


# ─────────────────────────────────────────────
# 5. API ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main HTML frontend"""
    return send_from_directory("static", "index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """Main generation endpoint"""
    data = request.get_json()

    user_input = data.get("input", "").strip()
    doc_type = data.get("doc_type", "patient_summary")

    if not user_input:
        return jsonify({"success": False, "error": "Input cannot be empty."}), 400

    if doc_type not in PROMPT_TEMPLATES:
        return jsonify({"success": False, "error": "Invalid document type."}), 400

    try:
        result = generate_healthcare_content(user_input, doc_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Fetch all stored documents from Vector DB"""
    try:
        all_docs = collection.get()
        history = []
        for i, doc_id in enumerate(all_docs["ids"]):
            history.append({
                "id": doc_id,
                "metadata": all_docs["metadatas"][i],
                "preview": all_docs["documents"][i][:200] + "..."
            })
        return jsonify({"success": True, "count": len(history), "history": history})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return Vector DB statistics"""
    try:
        count = collection.count()
        return jsonify({"success": True, "total_stored": count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────────
# 6. RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Healthcare GenAI Tool — Running on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
