# 🏥 MedScribe AI — Healthcare GenAI Content Generation Tool

## Project Overview
A specialized GenAI tool for healthcare professionals that transforms basic case notes into professional clinical documents using **Prompt Engineering**, **Vector DB (RAG)**, and **Gemini LLM API**.

---

## Tech Stack
| Layer | Tool |
|-------|------|
| Backend | Python + Flask |
| LLM API | Google Gemini 1.5 Flash (Free Tier) |
| Vector DB | ChromaDB (in-memory / persistent) |
| Frontend | HTML + CSS + Vanilla JS |

---

## How It Works (Pipeline)

```
User Input → Vector DB Retrieval → Prompt Engineering → Gemini LLM → Output + Store in DB
```

1. **User Input**: Doctor enters a brief patient description or topic
2. **Vector DB Retrieval (RAG)**: ChromaDB is queried for similar past documents — this provides context
3. **Prompt Engineering**: A carefully crafted template injects the context + user input with clinical instructions for tone, format, and terminology
4. **Gemini LLM**: The engineered prompt is sent to Gemini 1.5 Flash (free tier)
5. **Output + Store**: The result is displayed and stored in ChromaDB for future retrieval

---

## Setup & Run

### 1. Get a Free Gemini API Key
- Visit: https://aistudio.google.com/app/apikey
- Create a free API key

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Your API Key
In `app.py`, replace:
```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```
Or set it as an environment variable:
```bash
export GEMINI_API_KEY="your_key_here"
```

### 4. Run the App
```bash
python app.py
```

### 5. Open in Browser
```
http://localhost:5000
```

---

## Document Types Supported
- **Patient Summary** — SOAP-style structured summary
- **Discharge Note** — Hospital discharge documentation
- **Clinical Referral** — Specialist referral letter
- **Medical Report** — Formal diagnostic report

---

## Project Structure
```
healthcare-genai/
├── app.py              ← Main backend (Flask + Gemini + ChromaDB)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── static/
    └── index.html      ← Frontend UI
```

---

## Key Concepts Demonstrated
- **Prompt Engineering**: Role-based system prompts with format constraints and clinical terminology enforcement
- **RAG (Retrieval-Augmented Generation)**: ChromaDB stores and retrieves similar past documents to enrich prompts
- **Vector Embeddings**: Documents are embedded and stored for semantic similarity search
- **LLM API Integration**: Gemini 1.5 Flash via Google's official Python SDK
