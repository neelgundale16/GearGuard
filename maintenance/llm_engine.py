"""
GearGuard LLM Engine
- Zero Ollama dependency (works on Render)
- Uses Google Gemini free tier if GEMINI_API_KEY env var is set
- Graceful rule-based fallback if no key present (zero config needed)
"""

import json
import os

import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-1.5-flash:generateContent"
)


def _call_gemini(prompt: str) -> str | None:
    """Call Gemini Flash (free tier). Returns text or None on any failure."""
    if not GEMINI_API_KEY:
        return None
    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None


def get_maintenance_insight(
    equipment_name: str, days_overdue: int, failure_history: list
) -> dict:
    """
    Return maintenance insight dict with keys:
        recommendation (str), risk_level (str: Low/Medium/High/Critical)
    Works with or without GEMINI_API_KEY.
    """
    if GEMINI_API_KEY:
        recent = ", ".join(failure_history[-3:]) if failure_history else "None"
        prompt = (
            f"Equipment: {equipment_name}\n"
            f"Days overdue for maintenance: {days_overdue}\n"
            f"Recent failures: {recent}\n"
            f"Provide a 2-sentence maintenance recommendation and a risk level "
            f"(Low/Medium/High/Critical). "
            f'Respond ONLY in JSON: {{"recommendation": "...", "risk_level": "..."}}'
        )
        raw = _call_gemini(prompt)
        if raw:
            try:
                clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                return json.loads(clean)
            except Exception:
                pass

    # Rule-based fallback — always works, no API key needed
    if days_overdue > 30:
        risk = "Critical"
        rec  = (
            f"{equipment_name} is severely overdue for maintenance. "
            "Schedule an emergency inspection immediately to prevent failure."
        )
    elif days_overdue > 14:
        risk = "High"
        rec  = (
            f"{equipment_name} is overdue. "
            "Schedule maintenance within the next 48 hours."
        )
    elif days_overdue > 0:
        risk = "Medium"
        rec  = (
            f"{equipment_name} maintenance is due. "
            "Plan and execute within this week."
        )
    else:
        risk = "Low"
        rec  = (
            f"{equipment_name} is within its maintenance schedule. "
            "Continue standard monitoring."
        )

    return {"recommendation": rec, "risk_level": risk}


def get_anomaly_explanation(anomaly_data: dict) -> str:
    """Explain a detected anomaly. Falls back to rule-based if no API key."""
    if GEMINI_API_KEY:
        prompt = (
            f"Maintenance anomaly detected: {json.dumps(anomaly_data)}\n"
            "Explain in one sentence why this is unusual and what action to take."
        )
        result = _call_gemini(prompt)
        if result:
            return result.strip()

    req_type = anomaly_data.get("request_type", "unknown")
    priority = anomaly_data.get("priority", "unknown")
    return (
        f"Unusual {priority}-priority {req_type} request detected outside normal patterns. "
        "Review scheduling and allocate resources accordingly."
    )


def generate_monthly_summary(stats: dict) -> str:
    """Generate executive monthly summary. Falls back to rule-based if no API key."""
    if GEMINI_API_KEY:
        prompt = (
            f"Monthly maintenance stats: {json.dumps(stats)}\n"
            "Write a 3-sentence executive summary highlighting key trends and recommendations."
        )
        result = _call_gemini(prompt)
        if result:
            return result.strip()

    total     = stats.get("total_requests", 0)
    completed = stats.get("completed", 0)
    critical  = stats.get("critical", 0)
    rate      = round((completed / total * 100) if total else 0, 1)

    trend_note = (
        "Performance is on track — maintain current scheduling cadence."
        if rate > 80
        else "Completion rate requires improvement — review resource allocation and backlog."
    )

    return (
        f"This month recorded {total} maintenance requests with a {rate}% completion rate. "
        f"{critical} critical issues required immediate escalation and intervention. "
        f"{trend_note}"
    )