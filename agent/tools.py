"""
agent/tools.py
==============
LangChain Tool definitions for the Healthcare AI Agent.

Each function is decorated with LangChain's @tool decorator which converts
it into a proper LangChain BaseTool object. This means:
  - The AgentExecutor in executor.py can call these autonomously
  - The agent reads each docstring to decide when to use each tool
  - No manual tool lookup or TOOL_REGISTRY dict is needed

LangChain @tool contract:
  - Input  : single string parameter
  - Output : string (the agent reads it as an "observation")
  - Docstring : tells the agent WHEN and HOW to use the tool
"""

import random
from datetime import datetime, timedelta
from langchain.tools import tool


# ── Mock Data ──────────────────────────────────────────────────────────────────
SPECIALISTS = {
    "cardiologist":       {"name": "Dr. Priya Mehta",      "available": True,  "next_slot": 2},
    "neurologist":        {"name": "Dr. Arjun Rao",        "available": True,  "next_slot": 4},
    "endocrinologist":    {"name": "Dr. Sunita Kapoor",    "available": True,  "next_slot": 1},
    "oncologist":         {"name": "Dr. Vikram Nair",      "available": False, "next_slot": 7},
    "pulmonologist":      {"name": "Dr. Ananya Sharma",    "available": True,  "next_slot": 3},
    "gastroenterologist": {"name": "Dr. Rahul Desai",      "available": True,  "next_slot": 5},
    "general physician":  {"name": "Dr. Meena Iyer",       "available": True,  "next_slot": 0},
    "orthopedist":        {"name": "Dr. Suresh Pillai",    "available": True,  "next_slot": 2},
    "psychiatrist":       {"name": "Dr. Kavita Bose",      "available": True,  "next_slot": 6},
    "dermatologist":      {"name": "Dr. Neel Verma",       "available": True,  "next_slot": 3},
}

EQUIPMENT = {
    "mri scanner":    {"available": True,  "wait_hours": 4,  "location": "Radiology Wing B"},
    "ct scanner":     {"available": True,  "wait_hours": 2,  "location": "Radiology Wing A"},
    "ecg machine":    {"available": True,  "wait_hours": 0,  "location": "Cardiology OPD"},
    "echocardiogram": {"available": False, "wait_hours": 24, "location": "Cardiology ICU"},
    "x-ray":          {"available": True,  "wait_hours": 1,  "location": "Radiology Wing A"},
    "blood lab":      {"available": True,  "wait_hours": 1,  "location": "Pathology Lab 1"},
    "ultrasound":     {"available": True,  "wait_hours": 3,  "location": "Radiology Wing C"},
    "spirometer":     {"available": True,  "wait_hours": 0,  "location": "Pulmonology OPD"},
    "endoscope":      {"available": False, "wait_hours": 48, "location": "GI Suite"},
    "defibrillator":  {"available": True,  "wait_hours": 0,  "location": "Emergency Ward"},
}

HISTORIES = {
    "diabetes":         "Patient has Type 2 Diabetes (diagnosed 2018). HbA1c: 8.9%. On Metformin 500mg.",
    "hypertension":     "BP consistently above 140/90 mmHg over 6 months. On Amlodipine 5mg.",
    "cardiac":          "No prior cardiac events. Family history: CAD (father). ECG 2 years ago — normal.",
    "respiratory":      "Mild asthma since childhood. Salbutamol inhaler PRN. FEV1 78% predicted.",
    "neurological":     "No prior neurological diagnoses. Occasional migraines. No seizure history.",
    "orthopedic":       "Prior right knee arthroscopy (2020). Physiotherapy completed.",
    "cancer":           "No personal malignancy history. Mother: breast cancer (survivor).",
    "gastrointestinal": "History of GERD. On PPI. Colonoscopy 3 years ago: normal.",
}

DURATIONS = {
    "medication adjustment":     {"days": 30,  "intensity": "Low",    "followup": "4 weeks"},
    "physiotherapy":             {"days": 21,  "intensity": "Medium", "followup": "Weekly"},
    "surgery":                   {"days": 7,   "intensity": "High",   "followup": "2 weeks"},
    "chemotherapy":              {"days": 180, "intensity": "High",   "followup": "3 weeks"},
    "radiation therapy":         {"days": 42,  "intensity": "High",   "followup": "Weekly"},
    "lifestyle modification":    {"days": 90,  "intensity": "Low",    "followup": "Monthly"},
    "diagnostic workup":         {"days": 7,   "intensity": "Medium", "followup": "After results"},
    "consultation":              {"days": 1,   "intensity": "Low",    "followup": "As needed"},
    "blood pressure monitoring": {"days": 30,  "intensity": "Low",    "followup": "2 weeks"},
    "insulin therapy":           {"days": 365, "intensity": "Medium", "followup": "Monthly"},
}


# ── LangChain @tool decorated functions ───────────────────────────────────────
# The @tool decorator converts each function into a LangChain StructuredTool.
# The AgentExecutor reads the docstring to decide when to call each tool.
# Return type must be a plain string — the agent uses it as its observation.

@tool
def check_specialist(specialist_type: str) -> str:
    """
    Check if a medical specialist is available for consultation.
    Use this when a step requires verifying doctor availability.
    Input: specialist type such as cardiologist, endocrinologist, neurologist.
    """
    key     = specialist_type.lower().strip()
    matched = next((k for k in SPECIALISTS if k in key or key in k), None)

    if matched:
        info      = SPECIALISTS[matched]
        slot_date = (datetime.now() + timedelta(days=info["next_slot"])).strftime("%d %b %Y")
        if info["available"]:
            return f"AVAILABLE: {info['name']} | Next slot: {slot_date} | Wait: {info['next_slot']} days"
        else:
            return f"UNAVAILABLE: {info['name']} | Estimated wait: {info['next_slot']} days"

    return f"AVAILABLE: A {specialist_type} can be arranged. Contact scheduling. Wait: 3 days"


@tool
def check_equipment(equipment_name: str) -> str:
    """
    Check if medical equipment or a diagnostic facility is available.
    Use this when a step requires verifying diagnostic equipment availability.
    Input: equipment name such as MRI scanner, ECG machine, blood lab, CT scanner.
    """
    key     = equipment_name.lower().strip()
    matched = next((k for k in EQUIPMENT if k in key or key in k), None)

    if matched:
        info = EQUIPMENT[matched]
        if info["available"]:
            return f"AVAILABLE: {matched.title()} at {info['location']} | Wait: ~{info['wait_hours']} hour(s)"
        else:
            return f"UNAVAILABLE: {matched.title()} at {info['location']} | Expected wait: {info['wait_hours']} hours"

    return f"AVAILABLE: {equipment_name} can be scheduled. Contact diagnostic centre. Wait: ~2 hours"


@tool
def check_insurance(procedure: str) -> str:
    """
    Check insurance coverage for a medical procedure or treatment.
    Use this when a step requires verifying whether a procedure is covered.
    Input: procedure name such as MRI scan, specialist consultation, blood test.
    """
    covered = ["blood test", "x-ray", "ecg", "consultation", "mri", "ct scan",
               "ultrasound", "general checkup", "vaccination", "physiotherapy"]
    partial = ["specialist visit", "dental", "vision", "cosmetic"]
    key     = procedure.lower()

    if any(p in key for p in covered):
        copay = random.choice([0, 200, 500])
        return f"COVERED: {procedure} | Co-pay: Rs.{copay} | Provider: General Insurance"
    elif any(p in key for p in partial):
        copay = random.choice([1000, 1500, 2000])
        return f"PARTIAL COVERAGE: {procedure} | Co-pay: Rs.{copay} | Provider: General Insurance"
    else:
        return f"NOT COVERED: {procedure} may not be covered. Patient bears full cost."


@tool
def get_patient_history(condition: str) -> str:
    """
    Retrieve patient medical history relevant to a specific condition.
    Use this when a step requires gathering background medical information.
    Input: condition name such as diabetes, hypertension, cardiac, respiratory.
    """
    key     = condition.lower()
    matched = next((k for k in HISTORIES if k in key or key in k), None)
    return HISTORIES.get(matched, f"No prior history found for {condition}. Patient record appears clean.")


@tool
def estimate_duration(treatment: str) -> str:
    """
    Estimate the duration and intensity of a treatment or intervention.
    Use this when a step requires planning timelines for a treatment plan.
    Input: treatment name such as physiotherapy, medication adjustment, surgery.
    """
    key     = treatment.lower()
    matched = next((k for k in DURATIONS if k in key or key in k), None)

    if matched:
        info = DURATIONS[matched]
        return (f"Duration: {info['days']} days | Intensity: {info['intensity']} | "
                f"Follow-up: {info['followup']}")

    return f"Duration for {treatment}: estimated 2-4 weeks | Intensity: Medium | Follow-up: As needed"


# ── LangChain Tool List ────────────────────────────────────────────────────────
# Passed directly to AgentExecutor in executor.py
# Each item is a LangChain BaseTool object (created by @tool decorator)
LANGCHAIN_TOOLS = [
    check_specialist,
    check_equipment,
    check_insurance,
    get_patient_history,
    estimate_duration,
]