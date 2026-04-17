"""
src/llm_analyst.py — FootballIQ LLM Reasoning Engine
======================================================
Sends the RAG context + blended probabilities to Claude claude-sonnet-4-20250514
and returns a structured JSON match briefing.

Expected JSON output shape:
{
  "predicted_outcome":  "Home Win | Draw | Away Win",
  "confidence":         "High | Medium | Low",
  "key_factors":        ["factor1", "factor2", "factor3"],
  "risk_flags":         ["risk1", "risk2"],
  "reasoning_summary":  "2-3 sentence plain English explanation.",
  "elo_probability":    {"home": 0.xx, "draw": 0.xx, "away": 0.xx}
}
"""

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Lazy-load Anthropic so the app works without the key installed ────────────
_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in environment / .env file.")
        _client = anthropic.Anthropic(api_key=api_key)
        return _client
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")


SYSTEM_PROMPT = """You are a professional football analyst with 20+ years of experience covering
European football. You are precise, data-driven, and honest about uncertainty.

Given pre-match statistical context and a probability estimate from a machine-learning model,
your task is to produce a structured pre-match briefing in valid JSON — nothing else.

The JSON must have EXACTLY these keys:
{
  "predicted_outcome":  "Home Win" | "Draw" | "Away Win",
  "confidence":         "High" | "Medium" | "Low",
  "key_factors":        [list of 2-4 strings explaining the main deciding factors],
  "risk_flags":         [list of 1-3 strings describing upset risks or uncertainties],
  "reasoning_summary":  "2-3 sentence plain-English explanation of your verdict.",
  "elo_probability":    {"home": <float 0-1>, "draw": <float 0-1>, "away": <float 0-1>}
}

Rules:
- Output ONLY the JSON object — no markdown fences, no extra text.
- Confidence is "High" if the model probability for the top outcome > 0.55, 
  "Medium" if 0.40-0.55, "Low" if < 0.40.
- Base your reasoning on the statistical context provided; do not hallucinate player names.
- If injury data says "Data unavailable", note the uncertainty in risk_flags.
"""

USER_PROMPT_TEMPLATE = """{rag_context}

--- MACHINE LEARNING PROBABILITIES ---
Blended model probabilities (60% XGB / 40% Elo):
  Home Win : {p_home:.1%}
  Draw     : {p_draw:.1%}
  Away Win : {p_away:.1%}
Top model prediction: {top_outcome}

Produce your structured JSON briefing now."""


def analyse(
    rag_context: str,
    blended_probs: dict,
    model_name: str = "claude-sonnet-4-20250514",
    max_tokens: int = 800,
) -> dict:
    """
    Call Claude claude-sonnet-4-20250514 with the RAG context and return a parsed JSON dict.

    Parameters
    ----------
    rag_context    : Natural-language pre-match context block.
    blended_probs  : {"home": float, "draw": float, "away": float}
    model_name     : Anthropic model ID.
    max_tokens     : Max response tokens.

    Returns
    -------
    dict — parsed JSON briefing, or a fallback dict on error.
    """
    p_home  = blended_probs.get("home", 0.0)
    p_draw  = blended_probs.get("draw", 0.0)
    p_away  = blended_probs.get("away", 0.0)
    top_map = {p_home: "Home Win", p_draw: "Draw", p_away: "Away Win"}
    top_outcome = top_map[max(p_home, p_draw, p_away)]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        rag_context=rag_context,
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
        top_outcome=top_outcome,
    )

    try:
        client   = _get_client()
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = response.content[0].text.strip()

        # ── Parse JSON ────────────────────────────────────────────────────
        try:
            # Strip accidental markdown fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            result = json.loads(raw_text)
            # Ensure elo_probability is populated
            if "elo_probability" not in result:
                result["elo_probability"] = blended_probs
            return result

        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON. Storing raw text as reasoning.")
            return _fallback_response(blended_probs, top_outcome, raw_text)

    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        return _fallback_response(
            blended_probs,
            top_outcome,
            f"LLM analysis unavailable: {str(exc)[:200]}",
        )


def _fallback_response(blended_probs: dict, top_outcome: str, raw_text: str) -> dict:
    """Return a structured fallback if the LLM call or JSON parse fails."""
    p_home = blended_probs.get("home", 0.0)
    p_draw = blended_probs.get("draw", 0.0)
    p_away = blended_probs.get("away", 0.0)
    top_p  = max(p_home, p_draw, p_away)
    confidence = "High" if top_p > 0.55 else ("Medium" if top_p > 0.40 else "Low")

    return {
        "predicted_outcome": top_outcome,
        "confidence":        confidence,
        "key_factors":       ["Based on blended XGB + Elo model probabilities"],
        "risk_flags":        ["LLM analysis unavailable — using model-only prediction"],
        "reasoning_summary": raw_text[:500] if raw_text else
                             "Analysis could not be generated. The model-based probabilities are shown above.",
        "elo_probability":   blended_probs,
        "_llm_error":        True,
    }
