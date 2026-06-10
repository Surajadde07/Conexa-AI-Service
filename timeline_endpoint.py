"""
timeline_endpoint.py

Add this endpoint to your existing FastAPI main.py.

It accepts a transcript and returns a structured conversation timeline
with timestamps and section titles — ready for the frontend TimelinePanel.

Install requirement (already in most setups):
    pip install ollama   # or use the existing LLM client in your service

─── Request ────────────────────────────────────────────────────────────────
POST /timeline
Content-Type: application/json
{
  "transcript": "Agent: Hello, thank you for calling..."
}

─── Response ───────────────────────────────────────────────────────────────
[
  { "time": "00:00", "title": "Greeting" },
  { "time": "01:25", "title": "Customer Problem" },
  { "time": "03:40", "title": "Solution Discussion" },
  { "time": "06:15", "title": "Payment Discussion" },
  { "time": "08:30", "title": "Call Closure" }
]
"""

from fastapi import APIRouter
from pydantic import BaseModel
import json, re, requests

router = APIRouter()


class TimelineRequest(BaseModel):
    transcript: str


# ── Estimated reading speed (words per minute) ──────────────────────────────
WPM = 150


def words_to_timestamp(word_index: int) -> str:
    """Convert a word index to a MM:SS timestamp assuming WPM reading speed."""
    total_seconds = int((word_index / WPM) * 60)
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"


def build_fallback_timeline(transcript: str) -> list[dict]:
    """
    Pure-Python fallback timeline — no LLM required.
    Scans the transcript for conversation-phase keywords and returns
    timestamped sections based on estimated 150 wpm reading speed.
    """
    PHASES = [
        {"keywords": ["hello","hi","good morning","good afternoon","welcome","how can i help"],
         "title": "Greeting"},
        {"keywords": ["problem","issue","trouble","not working","error","complaint","concern","unable","calling because"],
         "title": "Customer Problem"},
        {"keywords": ["let me check","looking into","verify","searching your account","one moment"],
         "title": "Investigation"},
        {"keywords": ["i can help","solution","we can","offer you","provide","discount","waive","resolve"],
         "title": "Solution Discussion"},
        {"keywords": ["payment","billing","charge","invoice","refund","credit"],
         "title": "Payment / Billing"},
        {"keywords": ["escalat","transfer","supervisor","manager","specialist"],
         "title": "Escalation"},
        {"keywords": ["anything else","is there anything","all set","satisfied","thank you for calling","have a great"],
         "title": "Call Closure"},
    ]

    words = transcript.split()
    timeline = []
    last_wi = -1

    for phase in PHASES:
        for wi in range(len(words)):
            chunk = " ".join(words[wi : wi + 8]).lower()
            if any(kw in chunk for kw in phase["keywords"]):
                if wi > last_wi + 30:
                    timeline.append({"time": words_to_timestamp(wi), "title": phase["title"]})
                    last_wi = wi
                    break

    # Always add an opening entry
    if not timeline or timeline[0]["time"] != "00:00":
        timeline.insert(0, {"time": "00:00", "title": "Opening"})

    return timeline


@router.post("/timeline")
async def generate_timeline(req: TimelineRequest):
    """
    Generate a conversation timeline from a transcript.

    Tries Ollama first (same model your service already uses).
    Falls back to the keyword heuristic if Ollama is unavailable or returns
    unparseable output.
    """
    transcript = req.transcript.strip()
    if not transcript:
        return []

    prompt = f"""You are a call center analytics AI.

Analyze this customer service call transcript and return a JSON array of
conversation phases with their approximate timestamps.

Use these phase names (use only the phases that actually appear):
- Greeting
- Customer Problem  
- Investigation
- Solution Discussion
- Payment / Billing
- Objection Handling
- Escalation
- Call Closure

Estimate timestamps based on the proportion of the transcript. The call
starts at 00:00. If the transcript looks like a 10-minute call, distribute
the timestamps accordingly.

Return ONLY a JSON array, no markdown, no explanation:
[
  {{"time": "00:00", "title": "Greeting"}},
  {{"time": "01:30", "title": "Customer Problem"}}
]

TRANSCRIPT:
{transcript[:4000]}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:latest",   # same model already used in your service
                "prompt": prompt,
                "stream": False,
            },
            timeout=30,
        )
        raw = response.json().get("response", "")

        # Extract JSON array from response (strip any markdown fencing)
        raw_clean = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\[.*\]", raw_clean, re.DOTALL)
        if match:
            timeline = json.loads(match.group(0))
            # Validate structure
            if isinstance(timeline, list) and all("time" in t and "title" in t for t in timeline):
                # Ensure 00:00 entry exists
                if not timeline or timeline[0]["time"] != "00:00":
                    timeline.insert(0, {"time": "00:00", "title": "Opening"})
                return timeline

    except Exception:
        pass  # Fall through to heuristic

    # Heuristic fallback — always succeeds
    return build_fallback_timeline(transcript)


# ─── HOW TO WIRE THIS INTO YOUR EXISTING main.py ─────────────────────────────
#
# Option A: paste the @router.post("/timeline") function directly into main.py
#
# Option B: include this file as a module:
#
#   from timeline_endpoint import router as timeline_router
#   app.include_router(timeline_router)
#
# ─────────────────────────────────────────────────────────────────────────────
