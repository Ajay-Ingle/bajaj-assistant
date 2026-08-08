"""Session-scoped account selection helper.

Session state itself lives in Streamlit's st.session_state (see
chat_ui/app.py) -- Streamlit already provides per-user session storage
natively, so the process-local `sessions` dict + generated `session_id`
layer from the earlier FastAPI-shaped design has been removed. Every
orchestrator function now takes the session dict directly and mutates it
in place, the same dict Streamlit owns.
"""


def select_account(session: dict, lan: str) -> None:
    """Set the selected loan account on a session dict, in place."""
    session["selected_lan"] = lan
