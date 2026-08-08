"""Customer-facing chat + voice UI (Streamlit) -- pure in-process, no HTTP.

Imports app/loan, app/crosssell, app/funds, app/orchestrator directly and
calls them as plain Python functions in this same process -- no FastAPI,
no `requests`, no localhost port for a backend to talk to. Two
independent Streamlit apps run side by side: this one (chat, port 8501)
and dashboard/app.py (Stage 2's internal SHAP/propensity dashboard, port
8502) -- neither talks to the other; they just both import from `app/`.

Never shows crosssell_result or SHAP values -- same customer-facing
boundary as always (dashboard-only).

Run with: streamlit run chat_ui/app.py --server.port 8501
"""

import io
import sys
from pathlib import Path

import streamlit as st
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.crosssell.score import score_customer  # noqa: E402
from app.loan.tools import list_loan_accounts  # noqa: E402
from app.orchestrator.llm_client import get_groq_client, handle_turn  # noqa: E402
from app.orchestrator.session import select_account  # noqa: E402

st.set_page_config(page_title="Bajaj Finserv Assistant", page_icon="\U0001F4AC")
st.title("Bajaj Finserv Assistant")

if "session" not in st.session_state:
    st.session_state.session = {
        "mobile_no": None,
        "selected_lan": None,
        "crosssell_result": None,
        "already_pitched": False,
        "history": [],
    }
if "stage" not in st.session_state:
    st.session_state.stage = "mobile_entry"  # "mobile_entry" | "account_select" | "chat"
if "last_turn_meta" not in st.session_state:
    st.session_state.last_turn_meta = None
if "auto_speak" not in st.session_state:
    st.session_state.auto_speak = True


def _transcribe_audio(audio_bytes: bytes) -> str:
    client = get_groq_client()
    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=("recording.wav", audio_bytes),
    )
    return transcription.text


def _speak(text: str) -> None:
    """Generate and play TTS audio for `text`, degrading quietly on
    failure (e.g. no internet -- gTTS calls out to a Google endpoint)."""
    try:
        tts = gTTS(text=text, lang="en")
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        st.audio(buffer, format="audio/mp3", autoplay=True)
    except Exception:
        st.caption("voice playback unavailable")


def send_message(text: str) -> None:
    """Single entry point for a turn -- used by both the typed and voice
    input paths so response handling is never duplicated between them."""
    with st.spinner("Thinking..."):
        try:
            result = handle_turn(st.session_state.session, text)
        except Exception as exc:
            # handle_turn only appends to history on success, so on
            # failure we append a visibly-distinct error turn ourselves
            # instead of crashing the whole app (rate limit, network,
            # bad key, etc.).
            st.session_state.session["history"].append({"role": "user", "content": text})
            st.session_state.session["history"].append({
                "role": "assistant",
                "content": f"⚠️ Sorry, something went wrong answering that: {exc}",
            })
            st.rerun()
            return

    st.session_state.last_turn_meta = {
        "token_usage": result["token_usage"],
        "latency_ms": result["latency_ms"],
        "tools_called": result["tools_called"],
    }

    if st.session_state.auto_speak:
        _speak(result["response"])

    st.rerun()


with st.sidebar:
    st.subheader("Dev info")
    st.caption(
        "Internal visibility only -- latency/token/tool info for this "
        "build, not a customer-facing feature."
    )
    if st.session_state.last_turn_meta:
        st.json(st.session_state.last_turn_meta)
    else:
        st.caption("No turns yet.")
    st.session_state.auto_speak = st.checkbox(
        "Auto-speak responses", value=st.session_state.auto_speak
    )


# --- screen 1: mobile number entry ---
if st.session_state.stage == "mobile_entry":
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
            st.session_state.session["mobile_no"] = mobile_no
            # None if the customer has no active loans -- score_customer()
            # already treats "not applicable" as None, not a crash.
            st.session_state.session["crosssell_result"] = score_customer(mobile_no)

            if len(accounts) == 1:
                select_account(st.session_state.session, accounts[0]["lan"])
                st.session_state.stage = "chat"
            else:
                st.session_state.accounts = accounts
                st.session_state.stage = "account_select"
            st.rerun()

# --- screen 2: account selection (only reached if there's more than one loan) ---
elif st.session_state.stage == "account_select":
    st.subheader("Select a loan account")
    accounts = st.session_state.accounts
    labels = [f"{a['lan']} — {a['loan_type']} ({a['loan_status']})" for a in accounts]
    choice = st.radio("Your loan accounts", options=labels, index=None)
    if choice is not None and st.button("Select this account"):
        lan = accounts[labels.index(choice)]["lan"]
        select_account(st.session_state.session, lan)
        st.session_state.stage = "chat"
        st.rerun()

# --- screen 3: chat ---
else:
    st.caption(f"Talking about loan account: {st.session_state.session['selected_lan']}")

    for message in st.session_state.session["history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    st.caption("Or use voice input:")
    audio = mic_recorder(
        start_prompt="\U0001F3A4 Start recording",
        stop_prompt="⏹️ Stop recording",
        just_once=True,
        key="recorder",
    )
    if audio is not None and audio.get("bytes"):
        with st.spinner("Transcribing..."):
            try:
                transcribed_text = _transcribe_audio(audio["bytes"])
            except Exception as exc:
                transcribed_text = None
                st.error(f"Voice transcription failed: {exc}")
        if transcribed_text:
            send_message(transcribed_text)

    user_message = st.chat_input("Ask about your loan or a Bajaj Finserv fund...")
    if user_message:
        send_message(user_message)
