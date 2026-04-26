"""
agent/executor.py
=================
Agent Execution Orchestrator — powered by LangChain AgentExecutor.

LangChain components used:
  - AgentExecutor          — runs the ReAct think -> tool -> observe loop
  - create_react_agent     — builds ReAct agent from llm + tools + prompt
  - ChatGroq — LangChain LLM for agent reasoning

app.py is NOT touched — it continues using raw google.generativeai directly.
Only the agent module uses LangChain.
"""

import json
import os

# ── LangChain imports ───────────
from langchain_classic.agents import AgentExecutor, create_react_agent

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from .planner import decompose_goal, build_schedule, replan_blocked_step
from .tools   import LANGCHAIN_TOOLS


Groq_API_KEY = os.environ.get("Groq_API_KEY", "YOUR_Groq_API_KEY_HERE")

# ── LangChain LLM for the Agent ────────────────────────────────────────────────
agent_llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # LLaMA 3 70B is highly recommended for ReAct agents
    api_key=Groq_API_KEY,
    temperature=0.1
)

# ── LangChain ReAct Prompt ─────────────────────────────────────────────────────
# Required placeholders for create_react_agent:
# {tools}, {tool_names}, {input}, {agent_scratchpad}
REACT_PROMPT = PromptTemplate.from_template(
    """You are a Healthcare AI Resource Validation Agent.
Check the availability of resources needed for a specific healthcare task.

Available tools:
{tools}

Tool names: {tool_names}

Use this EXACT format:
Thought: what tool should I use and why
Action: tool name (must be one of {tool_names})
Action Input: the input string for the tool
Observation: the tool result
... (repeat as needed)
Thought: I now have all information needed
Final Answer: clear summary of all findings

Task: {input}
{agent_scratchpad}"""
)

# ── LangChain AgentExecutor ────────────────────────────────────────────────────
react_agent = create_react_agent(
    llm=agent_llm,
    tools=LANGCHAIN_TOOLS,
    prompt=REACT_PROMPT
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=LANGCHAIN_TOOLS,
    verbose=False,
    max_iterations=4,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)


def _validate_step(step: dict) -> str:
    """
    Run resource validation for one step using LangChain AgentExecutor.
    The agent autonomously decides which tool to call based on the step description.
    Returns a plain string observation.
    """
    tool_name  = step.get("tool", "none")
    tool_param = step.get("tool_param", "")

    if tool_name == "none":
        return "No resource validation required for this step."

    task = (f"Check resource availability for this healthcare step: "
            f"'{step.get('title')}'. "
            f"Specifically check: {tool_param}. "
            f"Use the most appropriate tool.")

    try:
        result = agent_executor.invoke({"input": task})
        return result.get("output", "Resource check completed.")
    except Exception as e:
        # Fallback: call matching tool directly by name
        for lc_tool in LANGCHAIN_TOOLS:
            if lc_tool.name == tool_name:
                try:
                    return lc_tool.invoke(tool_param)
                except Exception:
                    pass
        return f"Resource check unavailable: {str(e)}"


# ── Full Pipeline (synchronous) ────────────────────────────────────────────────

def run_agent(goal: str) -> dict:
    """
    Full 4-phase agent pipeline using LangChain throughout.

    Phase 1: LangChain LLMChain     — decompose_goal()
    Phase 2: LangChain AgentExecutor — _validate_step() per step
    Phase 3: Pure Python             — build_schedule()
    Phase 4: LangChain LLMChain     — replan_blocked_step()
    """
    result = {"success": False, "goal": goal, "phases": {}}

    # Phase 1
    plan  = decompose_goal(goal)
    steps = plan.get("steps", [])

    result["phases"]["decomposition"] = {
        "steps_identified": len(steps),
        "steps":            steps,
        "summary":          plan.get("summary", "")
    }

    if not steps:
        result["error"] = "Agent could not decompose the goal."
        return result

    # Phase 2
    resource_results = []
    validation_log   = []

    for step in steps:
        obs   = _validate_step(step)
        resource_results.append(obs)
        is_ok = not any(w in obs.upper() for w in ["UNAVAILABLE", "NOT COVERED"])
        validation_log.append({
            "step_id":    step["id"],
            "step_title": step["title"],
            "tool_used":  step.get("tool", "none"),
            "resource_ok": is_ok,
            "details":    obs
        })

    result["phases"]["validation"] = validation_log

    # Phase 3
    schedule = build_schedule(plan, resource_results)
    result["phases"]["schedule"] = schedule

    # Phase 4
    replanning_events = []
    blocked = [s for s in schedule["scheduled_steps"] if s["status"] == "blocked"]

    for blocked_step in blocked:
        idx = next((i for i, s in enumerate(steps) if s["id"] == blocked_step["id"]), None)
        if idx is None:
            continue

        adaptation = replan_blocked_step(
            blocked_step,
            resource_results[idx],
            steps[idx + 1:]
        )
        replanning_events.append({
            "blocked_step":   blocked_step["title"],
            "adaptation":     adaptation.get("adaptation", ""),
            "modified_steps": adaptation.get("modified_steps", [])
        })

        for mod in adaptation.get("modified_steps", []):
            for sched_step in schedule["scheduled_steps"]:
                if sched_step["id"] == mod.get("id"):
                    sched_step["description"] = mod.get("description", sched_step["description"])
                    sched_step["status"]      = "adapted"

    if replanning_events:
        result["phases"]["replanning"] = replanning_events

    result["final_schedule"] = schedule
    result["summary"]        = plan.get("summary", f"Plan generated for: {goal}")
    result["success"]        = True
    return result


# ── Streaming Pipeline ─────────────────────────────────────────────────────────

def stream_agent(goal: str):
    """
    Generator version — yields SSE events as each phase completes.
    Uses LangChain throughout: LLMChain for decompose/replan,
    AgentExecutor for resource validation.
    """

    def event(phase, message, payload=None):
        data = {"phase": phase, "message": message}
        if payload:
            data["payload"] = payload
        return f"data: {json.dumps(data)}\n\n"

    yield event("start", f"LangChain Agent activated. Goal: '{goal}'")

    # Phase 1 — LangChain LLMChain
    yield event("decomposing", "LangChain LLMChain breaking down your goal...")
    plan  = decompose_goal(goal)
    steps = plan.get("steps", [])
    yield event("decomposed", f"Identified {len(steps)} steps.", {
        "steps": steps, "summary": plan.get("summary", "")
    })

    # Phase 2 — LangChain AgentExecutor
    yield event("validating", "LangChain AgentExecutor checking resources...")
    resource_results = []

    for step in steps:
        obs   = _validate_step(step)
        resource_results.append(obs)
        is_ok = not any(w in obs.upper() for w in ["UNAVAILABLE", "NOT COVERED"])
        yield event("resource_check",
                    f"Step {step['id']}: {step['title']} — {'available' if is_ok else 'unavailable'}", {
            "step_id":   step["id"],
            "tool":      step.get("tool", "none"),
            "available": is_ok,
            "details":   obs
        })

    # Phase 3
    yield event("scheduling", "Building dependency-aware execution schedule...")
    schedule = build_schedule(plan, resource_results)
    yield event("scheduled",
                f"Schedule ready. Duration: {schedule['total_days']} days.", {
        "schedule": schedule
    })

    # Phase 4 — LangChain LLMChain
    blocked = [s for s in schedule["scheduled_steps"] if s["status"] == "blocked"]
    if blocked:
        yield event("replanning",
                    f"{len(blocked)} blocked step(s). LangChain adapting plan...")
        for blocked_step in blocked:
            idx = next(
                (i for i, s in enumerate(steps) if s["id"] == blocked_step["id"]), None
            )
            if idx is None:
                continue
            adaptation = replan_blocked_step(
                blocked_step, resource_results[idx], steps[idx + 1:]
            )
            yield event("replanned", f"Adapted: {blocked_step['title']}", {
                "blocked":    blocked_step["title"],
                "adaptation": adaptation.get("adaptation", "")
            })

    yield event("complete", "LangChain Agent completed all phases.", {
        "final_schedule": schedule,
        "summary":        plan.get("summary", "")
    })