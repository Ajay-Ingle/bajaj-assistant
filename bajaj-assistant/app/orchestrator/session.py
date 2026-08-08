"""In-memory session store for the orchestrator (POC only).

A process-local dict doesn't survive a restart and doesn't work across
multiple app workers/instances -- production would back this with Redis
or a proper session store.

Sessions are keyed by a generated session_id (uuid4), never by raw mobile
number, so the frontend never has to resend the mobile number on every
request -- a deliberate PII-minimization choice.
"""

import uuid

from app.crosssell.score import score_customer

sessions: dict[str, dict] = {}
# each session: {
#     "mobile_no": str,
#     "selected_lan": str | None,
#     "crosssell_result": dict | None,   # from score_customer(), computed once
#     "already_pitched": bool,
#     "history": list[dict],             # running message history for this session
# }


def create_session(mobile_no: str) -> str:
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "mobile_no": mobile_no,
        "selected_lan": None,
        # None if the customer has no active loans -- score_customer()
        # already treats "not applicable" as None, not a crash or a 0.
        "crosssell_result": score_customer(mobile_no),
        "already_pitched": False,
        "history": [],
    }
    return session_id


def get_session(session_id: str) -> dict | None:
    return sessions.get(session_id)


def select_account(session_id: str, lan: str) -> bool:
    """Set the selected loan account for a session.

    Returns False if the session doesn't exist (caller decides how to
    respond, e.g. a 404), True on success.
    """
    session = sessions.get(session_id)
    if session is None:
        return False
    session["selected_lan"] = lan
    return True
