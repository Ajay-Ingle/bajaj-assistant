"""System prompt templates."""

from app.funds.fund_catalog import get_pitchable_fund

BASE_SYSTEM_PROMPT = """You are a voice-friendly assistant for Bajaj Finserv loan customers.

Rules you must follow:
- For loan questions, only answer using the get_loan_details tool result for the customer's selected loan account. Never invent, estimate, or recall loan numbers yourself -- always call the tool and use exactly what it returns.
- For mutual fund questions, only answer using the search_fund_factsheet tool result. If it returns found: False, tell the user that information isn't in the factsheet -- do not guess or use outside/general knowledge about mutual funds.
- Keep answers concise and conversational. Your response may be read aloud via text-to-speech, so avoid long lists, markdown formatting, or anything that reads awkwardly out loud.
"""


def build_system_prompt(session: dict, should_attempt_pitch: bool) -> str:
    """Build this turn's system prompt, appending a directive cross-sell
    instruction when `should_attempt_pitch` is True.

    `should_attempt_pitch` is computed once by the caller
    (app/orchestrator/llm_client.py:handle_turn) -- fixed to the session's
    first user message, so pitch timing is deterministic/demoable rather
    than left to "whenever the model feels it fits." The same flag is
    reused after generation there to run a deterministic verify-and-patch
    check, since a soft instruction alone isn't reliably followed.
    """
    prompt = BASE_SYSTEM_PROMPT

    if should_attempt_pitch:
        fund = get_pitchable_fund(session["crosssell_result"]["fund_tier"])
        prompt += (
            f"\nAfter answering the user's question, add one brief closing "
            f"sentence recommending {fund} to them, referencing their "
            f"strong repayment history as the reason. Phrase it naturally "
            f"and keep it to one sentence, but make sure this sentence is "
            f"included in your response."
        )

    return prompt
