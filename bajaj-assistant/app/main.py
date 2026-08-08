"""FastAPI app entrypoint.

The customer-facing chat flow (mobile-number session, account selection,
Groq-orchestrated chat) lives in frontend/chat_app.py (Streamlit) instead
of as FastAPI routes -- see app/orchestrator/session.py and
app/orchestrator/llm_client.py for the transport-agnostic logic Streamlit
calls directly, in-process. This file stays minimal, matching the
scaffold's original /health baseline.
"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}
