"""Customer-facing chat UI (Streamlit): loan Q&A + fund Q&A via the Groq orchestrator.

Customer-facing, unlike dashboard/app.py (Stage 2's internal-only model
verification tool) -- this screen never shows propensity scores or SHAP
values, only the assistant's natural-language response. The token
usage/latency/tools_called debug info handle_turn() returns is shown in a
collapsed expander per message for our own verification during this
build stage, not as a customer-facing feature.

Runs the orchestrator directly in-process (no FastAPI server/HTTP hop) --
app/orchestrator/session.py's module-level `sessions` dict is shared
within this one Streamlit process, keyed by a session_id that this page
tracks in st.session_state per browser tab, the same way an HTTP client
would track it across requests.

Run with: streamlit run frontend/chat_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.loan.tools import list_loan_accounts  # noqa: E402
from app.orchestrator.llm_client import handle_turn  # noqa: E402
from app.orchestrator.session import create_session, select_account  # noqa: E402

st.set_page_config(page_title="Bajaj Finserv Assistant", page_icon="\U0001F4AC")
st.title("Bajaj Finserv Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.accounts = None
    st.session_state.selected_lan = None
    st.session_state.chat_history = []  # list of (role, text, debug_info | None)


# --- Step 1: mobile number -> session + account list ---
# POC simplification: identity is established by mobile number alone, with
# no password/OTP verification -- there is no real authentication here.
if st.session_state.session_id is None:
    st.caption(
        "POC simplification: identity is established by mobile number "
        "alone, with no password/OTP verification."
    )
    mobile_no = st.text_input("Enter your registered mobile number")
    if st.button("Continue", disabled=not mobile_no):
        accounts = list_loan_accounts(mobile_no)
        if not accounts:
            st.error("No loan accounts found for this mobile number.")
        else:
            session_id = create_session(mobile_no)
            st.session_state.session_id = session_id
            st.session_state.accounts = accounts
            if len(accounts) == 1:
                select_account(session_id, accounts[0]["lan"])
                st.session_state.selected_lan = accounts[0]["lan"]
            st.rerun()

# --- Step 2: account selection (only shown if there's more than one loan) ---
elif st.session_state.selected_lan is None:
    st.subheader("Select a loan account")
    accounts = st.session_state.accounts
    labels = [f"{a['lan']} — {a['loan_type']} ({a['loan_status']})" for a in accounts]
    choice = st.radio("Your loan accounts", options=labels, index=None)
    if choice is not None and st.button("Select this account"):
        lan = accounts[labels.index(choice)]["lan"]
        select_account(st.session_state.session_id, lan)
        st.session_state.selected_lan = lan
        st.rerun()

# --- Step 3: chat ---
else:
    st.caption(f"Talking about loan account: {st.session_state.selected_lan}")

    for role, text, debug in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)
            if debug is not None:
                with st.expander("debug: token usage / latency / tools called"):
                    st.json(debug)

    user_message = st.chat_input("Ask about your loan or a Bajaj Finserv fund...")
    if user_message:
        st.session_state.chat_history.append(("user", user_message, None))
        with st.spinner("Thinking..."):
            result = handle_turn(st.session_state.session_id, user_message)
        debug_info = {
            "token_usage": result["token_usage"],
            "latency_ms": result["latency_ms"],
            "tools_called": result["tools_called"],
        }
        st.session_state.chat_history.append(("assistant", result["response"], debug_info))
        st.rerun()

    if st.button("Start over"):
        st.session_state.session_id = None
        st.session_state.accounts = None
        st.session_state.selected_lan = None
        st.session_state.chat_history = []
        st.rerun()
