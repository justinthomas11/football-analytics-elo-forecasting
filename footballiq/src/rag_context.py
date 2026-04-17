"""
src/rag_context.py — FootballIQ RAG Context Builder
=====================================================
Converts a scraper output dict into a clean natural-language
context block that gets injected into the LLM prompt.
"""

from typing import Optional


def _form_string(form_list: list) -> str:
    """Convert ['W','D','L','W','W'] → 'W D L W W' or 'No data'."""
    if not form_list:
        return "No data available"
    return " ".join(form_list)


def _goals_str(avg: Optional[float]) -> str:
    return f"{avg:.2f}" if avg is not None else "N/A"


def _h2h_summary(h2h: dict, home_team: str, away_team: str) -> str:
    hw     = h2h.get("home_wins", 0)
    aw     = h2h.get("away_wins", 0)
    draws  = h2h.get("draws", 0)
    total  = hw + aw + draws

    if total == 0:
        return "No head-to-head data available in dataset."

    lines = [
        f"Last {total} meetings: "
        f"{home_team} won {hw}, {away_team} won {aw}, {draws} draw(s)."
    ]

    matches = h2h.get("matches", [])
    if matches:
        recent = matches[-3:]  # show last 3 scorelines
        scorelines = ", ".join(f"{m['date'][:7]}: {m['score']}" for m in recent)
        lines.append(f"Recent scorelines: {scorelines}.")

    return " ".join(lines)


def build_rag_context(
    context: dict,
    home_team: str,
    away_team: str,
    home_elo: Optional[float] = None,
    away_elo: Optional[float] = None,
) -> str:
    """
    Build a structured natural-language context block from scraped data.

    Parameters
    ----------
    context    : dict returned by scraper.scrape_prematch_context()
    home_team  : display name
    away_team  : display name
    home_elo   : current Elo rating for home team (optional)
    away_elo   : current Elo rating for away team (optional)

    Returns
    -------
    str — the RAG context paragraph injected into the LLM prompt
    """
    home_form  = _form_string(context.get("home_form", []))
    away_form  = _form_string(context.get("away_form", []))
    hga        = _goals_str(context.get("home_goals_avg"))
    aga        = _goals_str(context.get("away_goals_avg"))
    hca        = _goals_str(context.get("home_conceded_avg"))
    aca        = _goals_str(context.get("away_conceded_avg"))
    h2h_text   = _h2h_summary(context.get("h2h_record", {}), home_team, away_team)
    data_source = context.get("source", "unknown")

    elo_block = ""
    if home_elo and away_elo:
        elo_block = (
            f"\nElo Ratings: {home_team} = {home_elo:.0f}, "
            f"{away_team} = {away_elo:.0f} "
            f"(Elo differential: {home_elo - away_elo:+.0f} in favour of "
            f"{'the home side' if home_elo >= away_elo else 'the away side'})."
        )

    block = f"""=== PRE-MATCH INTELLIGENCE REPORT ===

HOME TEAM: {home_team}
  • Last 5 results (oldest → newest): {home_form}
  • Avg goals scored (last 5): {hga}
  • Avg goals conceded (last 5): {hca}

AWAY TEAM: {away_team}
  • Last 5 results (oldest → newest): {away_form}
  • Avg goals scored (last 5): {aga}
  • Avg goals conceded (last 5): {aca}

HEAD-TO-HEAD:
  {h2h_text}
{elo_block}
(Data source: {data_source})
======================================"""

    return block.strip()
