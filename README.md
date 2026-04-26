# 🏥 MedScribe AI — Healthcare Agentic-GenAI Content Generation Tool

## Project Overview
A specialized Agentic-GenAI tool for healthcare professionals that transforms basic case notes into professional clinical documents and healthcare treatment goals into actionable step-by-step schedule card using **Prompt Engineering**, **Vector DB (RAG)**, **LangChain** and **Gemini LLM API**.

---

## Tech Stack
| Layer | Tool |
|-------|------|
| Backend | Python + Flask |
| LLM API | Google Gemini 2.5 Flash (Free Tier) |
|  | Groq API (AI Agent) |
| AI Agent | LangChain |
| Vector DB | ChromaDB (in-memory / persistent) |
| Frontend | HTML + CSS + Vanilla JS |

---

## How It Works (Pipeline)

```
User Input → Vector DB Retrieval → Prompt Engineering → Gemini LLM → Output + Store in DB
Agentic Workflow
User Input → Step Breakdown → Tool Validation → Step Updation → Groq → Output 
```

1. **User Input**: Doctor enters a brief patient description or topic
2. **Vector DB Retrieval (RAG)**: ChromaDB is queried for similar past documents — this provides context
3. **Prompt Engineering**: A carefully crafted template injects the context + user input with clinical instructions for tone, format, and terminology
4. **Gemini LLM**: The engineered prompt is sent to Gemini 2.5 Flash (free tier)
5. **Output + Store**: The result is displayed and stored in ChromaDB for future retrieval

Agentic 
1. **User Input**: Doctor enters a healthcare goal
2. **Step Breakdown**: The goal is then broken down into small actionable steps
3. **Tool Validation**: The steps so generated are then validated with available tools and resources
4. **Step Updation**: The steps so generated are then mark blocked and alternative found if resource unavailable otherwise continue with the generated steps
5. **Groq LLM**: The data is in constant touch with groq LLM for output final steps generation
6. **Output**: The result that is final schedule card is displayed on the interface
---

## Setup & Run

### 1. Get a Free Gemini API Key and Groq API Key
- Visit: https://aistudio.google.com/app/apikey for Gemini and https://console.groq.com/keys for Groq API
- Create a free API key

### 2. API key setup
- Create a .env file.
- Write GEMINI_API_KEY="your_actual_api_key"
- Write Groq_API_KEY="your_actual_api_key"
- Write your own API key in this section without "".

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Your API Key
After creating the .env file all points that require API key will fetch them directly

### 5. Run the App
```bash
python app.py
```

### 6. Open in Browser
```
http://localhost:5000
```

---

## Document Types Supported
- **Patient Summary** — SOAP-style structured summary
- **Medical Report** — Formal diagnostic report
- **Schedule Card** — The step-by-step procedure for providded healthcare goal

---

## Project Structure
```
MedScribe/
├── app.py              ← Main backend (Flask + Gemini + ChromaDB)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── agent/
    └── executor.py      ← Agentic Integration
    └── planner.py
    └── routes.py
    └── tools.py
    └── __init__.py
└── static/
    └── index.html      ← Frontend UI
    └── agent.html
```

---

## Key Concepts Demonstrated
- **Prompt Engineering**: Role-based system prompts with format constraints and clinical terminology enforcement
- **RAG (Retrieval-Augmented Generation)**: ChromaDB stores and retrieves similar past documents to enrich prompts
- **Vector Embeddings**: Documents are embedded and stored for semantic similarity search
- **LangChain Integration**: The agentic integration using LangChain for modular structure and effiicient
- **LLM API Integration**: Gemini 2.5 Flash via Google's official Python SDK and Groq via LangChain SDK