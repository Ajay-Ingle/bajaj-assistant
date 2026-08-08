"""System prompt templates."""

from app.funds.fund_catalog import get_pitchable_fund

BASE_SYSTEM_PROMPT = """You are a voice-friendly assistant for Bajaj Finserv loan customers.

Rules you must follow:
- For loan questions, only answer using the get_loan_details tool result for the customer's selected loan account. Never invent, estimate, or recall loan numbers yourself -- always call the tool and use exactly what it returns.
- For mutual fund questions, only answer using the search_fund_factsheet tool result. If it returns found: False, tell the user that information isn't in the factsheet -- do not guess or use outside/general knowledge about mutual funds.
- Keep answers concise and conversational. Your response may be read aloud via text-to-speech, so avoid long lists, markdown formatting, or anything that reads awkwardly out loud.
"""


def build_system_prompt(session: dict) -> tuple[str, bool]:
    """Build this turn's system prompt, appending a one-shot cross-sell
    nudge if the session's propensity score recommends a pitch and one
    hasn't been offered yet this session.

    Returns (system_prompt, cross_sell_instruction_included). The caller
    (app/orchestrator/llm_client.py) is responsible for setting
    session["already_pitched"] = True only after a response was actually
    generated using this prompt -- so a pitch that was offered but the
    turn failed doesn't get silently marked as used.
    """
    prompt = BASE_SYSTEM_PROMPT
    crosssell_result = session.get("crosssell_result")

    include_pitch = bool(
        crosssell_result
        and crosssell_result.get("pitch_recommended")
        and not session.get("already_pitched")
    )

    if include_pitch:
        fund = get_pitchable_fund(crosssell_result["fund_tier"])
        prompt += (
            f"\nIf it fits naturally in your response, you may mention that "
            f"{fund} could be worth exploring given the customer's strong "
            f"repayment profile -- keep it brief, one sentence, and don't "
            f"force it if the user's question has nothing to do with "
            f"savings or investing."
        )

    return prompt, include_pitch
