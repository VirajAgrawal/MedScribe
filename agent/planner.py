"""
agent/planner.py
================
Goal Decomposition & Schedule Generation using LangChain.

LangChain components used here:
  - ChatGroq  : LangChain LLM wrapper for Groq
  - PromptTemplate          : Structured prompt with declared input variables
  - LLMChain                : Links PromptTemplate + LLM declaratively

These replace what would otherwise be raw genai.GenerativeModel() calls
with manual string formatting — keeping this isolated entirely in the
agent module without touching app.py at all.
"""

import json
import re
import os

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


Groq_API_KEY = os.environ.get("Groq_API_KEY", "YOUR_Groq_API_KEY_HERE")

# ── LangChain LLM ──────────────────────────────────────────────────────────────
# ChatGroq is LangChain's wrapper for Groq.
# This is completely separate from app.py's genai.GenerativeModel() setup.
planner_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=Groq_API_KEY,
    temperature=0.2,
)

# ── LangChain PromptTemplate — Goal Decomposition ─────────────────────────────
# PromptTemplate declares input_variables explicitly and validates them.
# This replaces a plain Python f-string or .format() call.
decompose_prompt = PromptTemplate(
    input_variables=["goal"],
    template="""You are a Healthcare AI Planning Agent.
Break down the following healthcare goal into a structured JSON plan.

Return ONLY a valid JSON object (no markdown, no explanation):
{{
  "goal": "<original goal>",
  "summary": "<one sentence describing the overall plan>",
  "steps": [
    {{
      "id": 1,
      "title": "<short action title>",
      "description": "<what this step involves>",
      "tool": "<one of: check_specialist | check_equipment | check_insurance | get_patient_history | estimate_duration | none>",
      "tool_param": "<parameter for the tool>",
      "depends_on": [],
      "priority": "<high | medium | low>",
      "estimated_time": "<e.g. 1 day or 2 hours>",
      "category": "<research | specialist | diagnostic | insurance | treatment | followup>"
    }}
  ]
}}

Rules:
- Generate exactly 5 to 8 steps
- depends_on lists step IDs that must complete first
- Order: research -> history -> diagnostics -> specialist -> treatment -> followup
- Be specific with tool_param (exact specialist type or equipment name)

Goal: {goal}"""
)

# ── LangChain PromptTemplate — Dynamic Replanning ─────────────────────────────
replan_prompt = PromptTemplate(
    input_variables=["failed_step", "resource_result", "remaining_steps"],
    template="""You are a Healthcare AI Planning Agent. A resource is UNAVAILABLE.
Adapt the plan and return ONLY a JSON object (no markdown):

{{
  "adaptation": "<one sentence describing the change>",
  "modified_steps": [ {{ same step structure, adapted }} ]
}}

Failed step: {failed_step}
Resource result: {resource_result}
Remaining steps: {remaining_steps}

Suggest an alternative resource, adjust timing, or reorder steps intelligently."""
)

# ── LangChain LLMChains ────────────────────────────────────────────────────────
# LLMChain links a PromptTemplate to an LLM.
# Calling chain.invoke({"goal": "..."}) handles formatting + LLM call in one step.
decompose_chain = decompose_prompt | planner_llm
replan_chain    = replan_prompt | planner_llm


# ── JSON parser ────────────────────────────────────────────────────────────────
def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from LLM output."""
    if isinstance(text, dict):
        return text
    text = re.sub(r"```(?:json)?\s*", "", str(text)).strip().rstrip("`").strip()
    return json.loads(text)


# ── Public functions ───────────────────────────────────────────────────────────

def decompose_goal(goal: str) -> dict:
    """
    Phase 1 — Goal Decomposition via LangChain LLMChain.
    decompose_chain.invoke() fills the PromptTemplate and calls Groq.
    """
    result = decompose_chain.invoke({"goal": goal})
    raw    = result.content if hasattr(result, "content") else str(result)

    try:
        return _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {
            "goal":    goal,
            "summary": f"Comprehensive plan to address: {goal}",
            "steps": [{
                "id": 1, "title": "Initial Assessment",
                "description": "Conduct initial patient assessment.",
                "tool": "get_patient_history", "tool_param": goal,
                "depends_on": [], "priority": "high",
                "estimated_time": "1 day", "category": "research"
            }]
        }


def build_schedule(plan: dict, resource_results: list) -> dict:
    """
    Phase 3 — Dependency-aware schedule generation.
    Pure Python logic — no LangChain needed here.
    """
    steps       = plan.get("steps", [])
    scheduled   = []
    day_counter = {}

    for i, step in enumerate(steps):
        deps      = step.get("depends_on", [])
        start_day = max((day_counter.get(d, 0) for d in deps), default=0) if deps else 0
        duration  = _parse_duration(step.get("estimated_time", "1 day"))
        end_day   = start_day + duration
        day_counter[step["id"]] = end_day

        res         = resource_results[i] if i < len(resource_results) else {}
        res_text    = res if isinstance(res, str) else res.get("details", "")
        resource_ok = not any(w in res_text.upper() for w in ["UNAVAILABLE", "NOT COVERED"])

        scheduled.append({
            **step,
            "start_day":    start_day,
            "end_day":      end_day,
            "resource_ok":  resource_ok,
            "resource_info": res_text,
            "status":       "ready" if resource_ok else "blocked"
        })

    total_days = max((s["end_day"] for s in scheduled), default=7)
    return {
        "goal":          plan["goal"],
        "summary":       plan.get("summary", ""),
        "total_days":    total_days,
        "steps_count":   len(scheduled),
        "blocked_count": sum(1 for s in scheduled if s["status"] == "blocked"),
        "scheduled_steps": scheduled
    }


def replan_blocked_step(failed_step: dict, resource_result, remaining_steps: list) -> dict:
    """
    Phase 4 — Dynamic Replanning via LangChain LLMChain.
    replan_chain.invoke() fills the PromptTemplate and calls Groq.
    """
    result = replan_chain.invoke({
        "failed_step":     json.dumps(failed_step, indent=2),
        "resource_result": str(resource_result),
        "remaining_steps": json.dumps(remaining_steps, indent=2)
    })
    raw = result.content if hasattr(result, "content") else str(result)

    try:
        return _parse_json(raw)
    except Exception:
        return {
            "adaptation":    f"Alternative pathway for '{failed_step.get('title')}'. Manual scheduling recommended.",
            "modified_steps": remaining_steps
        }


def _parse_duration(estimated_time: str) -> int:
    et   = estimated_time.lower()
    nums = re.findall(r"\d+", et)
    n    = int(nums[0]) if nums else 1
    if "hour" in et:  return max(1, n // 8)
    if "week" in et:  return n * 7
    if "month" in et: return n * 30
    return n