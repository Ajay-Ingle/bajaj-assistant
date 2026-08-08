"""Wraps the Groq LLM client and tool-use wiring for the orchestrator.

Renamed from the scaffold's `claude_client.py` -- Stage 4 uses Groq
(llama-3.3-70b-versatile) rather than the Anthropic SDK. Groq's chat
completions API is OpenAI-compatible, so the tool-calling shape below
follows that convention (tools/tool_calls/role="tool"), not Anthropic's.

The model never reads the CSV or computes numbers itself: it calls
get_loan_details / search_fund_factsheet, gets an exact value back, and
only phrases that value into natural language. That's the core grounding
guarantee for real financial data -- raw customer rows never go into the
prompt.
"""

import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

from app.funds.retrieve import retrieve
from app.loan.tools import get_loan_details
from app.orchestrator.prompts import build_system_prompt

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_loan_details",
            "description": (
                "Get the exact, current details of the customer's selected "
                "loan account (EMI amount, tenure, months remaining, "
                "approximate outstanding balance, repayment status). "
                "Always call this for any question about the customer's "
                "loan -- never estimate or invent these numbers yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lan": {
                        "type": "string",
                        "description": (
                            "The loan account number (LAN). The server "
                            "scopes this to the customer's selected "
                            "account regardless of what is passed here."
                        ),
                    },
                },
                "required": ["lan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_fund_factsheet",
            "description": (
                "Search the Bajaj Finserv mutual fund factsheet for "
                "information about a fund's objective, risk, category, "
                "expense ratio, holdings, or other factsheet content. "
                "Always call this for any mutual fund question -- never "
                "answer fund questions from general/outside knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The fund-related question to search the factsheet for.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

_client = None


def get_groq_client() -> Groq:
    """Public so chat_ui/app.py can reuse it for Whisper transcription
    (client.audio.transcriptions.create) without duplicating the
    API-key-loading logic."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        _client = Groq(api_key=api_key)
    return _client


def _execute_tool_call(name: str, arguments: dict, session: dict) -> dict:
    """Run one tool call and return a JSON-serializable result.

    get_loan_details is always scoped to session["selected_lan"],
    regardless of what LAN the model passed -- the customer's account
    selection (see app.orchestrator.session.select_account) is the source
    of truth, not anything the LLM decides to look up.
    """
    if name == "get_loan_details":
        lan = session.get("selected_lan")
        if lan is None:
            return {"error": "no loan account selected for this session"}
        result = get_loan_details(lan)
        return result if result is not None else {"error": f"no loan found for LAN {lan}"}

    if name == "search_fund_factsheet":
        query = arguments.get("query", "")
        return retrieve(query)

    return {"error": f"unknown tool: {name}"}


def handle_turn(session: dict, user_message: str) -> dict:
    """Run one chat turn: build the prompt, call Groq, execute at most one
    round of tool calls, then return the final answer + usage/latency.

    Takes the session dict directly (owned by the caller, typically
    st.session_state.session in chat_ui/app.py) and mutates it in place
    (already_pitched, history) rather than returning a new dict to
    reassign -- there's no session store here to write back to.
    """
    client = get_groq_client()
    system_prompt, pitch_included = build_system_prompt(session)

    messages = (
        [{"role": "system", "content": system_prompt}]
        + session["history"]
        + [{"role": "user", "content": user_message}]
    )

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    tools_called = []
    assistant_message = response.choices[0].message

    if assistant_message.tool_calls:
        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in assistant_message.tool_calls
            ],
        })

        for tc in assistant_message.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}
            tool_result = _execute_tool_call(tool_name, tool_args, session)
            tools_called.append(tool_name)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result),
            })

        # Deliberately no `tools=` on this call -- caps tool-calling at one
        # round and forces a final natural-language answer instead of
        # letting the model chain further tool calls.
        follow_up = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            messages=messages,
        )
        final_text = follow_up.choices[0].message.content
        prompt_tokens += follow_up.usage.prompt_tokens
        completion_tokens += follow_up.usage.completion_tokens
        total_tokens += follow_up.usage.total_tokens
    else:
        final_text = assistant_message.content

    latency_ms = round((time.perf_counter() - start) * 1000)

    if pitch_included:
        session["already_pitched"] = True

    session["history"].append({"role": "user", "content": user_message})
    session["history"].append({"role": "assistant", "content": final_text})

    return {
        "response": final_text,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "latency_ms": latency_ms,
        "tools_called": tools_called,
    }
