"""
src/llm_analyst.py — FootballIQ LLM Reasoning Engine
======================================================
Sends RAG context + blended probabilities to Claude claude-sonnet-4-20250514
and returns a structured JSON match briefing.

Output JSON schema:
{
  "predicted_outcome":  "Home Win | Draw | Away Win",
  "confidence":         "High | Medium | Low",
  "key_factors":        ["factor1", "factor2", "factor3"],
  "risk_flags":         ["risk1"],
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

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found. Add it to your .env file."
            )
        _client = anthropic.Anthropic(api_key=api_key)
        return _client
    except ImportError:
        raise ImportError("Run: pip install anthropic")


SYSTEM_PROMPT = """You are a professional football analyst with 20+ years of experience
covering European football. You are precise, data-driven, and honest about uncertainty.

Given pre-match statistical context and ML-based probability estimates, produce a
structured pre-match briefing as a valid JSON object — nothing else.

The JSON must have EXACTLY these keys:
{
  "predicted_outcome":  "Home Win" | "Draw" | "Away Win",
  "confidence":         "High" | "Medium" | "Low",
  "key_factors":        [2-4 strings — the main deciding factors],
  "risk_flags":         [1-3 strings — upset risks or uncertainties],
  "reasoning_summary":  "2-3 sentence plain-English verdict.",
  "elo_probability":    {"home": <float>, "draw": <float>, "away": <float>}
}

Rules:
- Output ONLY the JSON object. No markdown fences, no extra text.
- Confidence: "High" if top outcome > 55%, "Medium" if 40-55%, "Low" if < 40%.
- Do not hallucinate player names. If data says unavailable, flag it in risk_flags.
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
    Call Claude and return a parsed JSON briefing dict.
    Falls back to a structured error dict — never raises.
    """
    p_home  = blended_probs.get("home", 0.0)
    p_draw  = blended_probs.get("draw", 0.0)
    p_away  = blended_probs.get("away", 0.0)
    top_map = {p_home: "Home Win", p_draw: "Draw", p_away: "Away Win"}
    top     = top_map[max(p_home, p_draw, p_away)]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        rag_context=rag_context,
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        top_outcome=top,
    )

    try:
        client   = _get_client()
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        if "elo_probability" not in result:
            result["elo_probability"] = blended_probs
        return result

    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON; wrapping as raw text.")
        return _fallback(blended_probs, top, raw)
    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        return _fallback(blended_probs, top,
                         f"LLM analysis unavailable: {str(exc)[:200]}")


def _fallback(blended_probs: dict, top_outcome: str, raw_text: str) -> dict:
    top_p = max(blended_probs.get("home", 0),
                blended_probs.get("draw", 0),
                blended_probs.get("away", 0))
    conf = "High" if top_p > 0.55 else ("Medium" if top_p > 0.40 else "Low")
    return {
        "predicted_outcome": top_outcome,
        "confidence":        conf,
        "key_factors":       ["Based on blended XGB + Elo model probabilities"],
        "risk_flags":        ["LLM analysis unavailable — model-only prediction shown"],
        "reasoning_summary": raw_text[:500] if raw_text else
                             "Analysis unavailable. Model-based probabilities are shown.",
        "elo_probability":   blended_probs,
        "_llm_error":        True,
    }
