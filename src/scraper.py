"""
src/scraper.py — FootballIQ Live Data Scraper
==============================================
Attempts to scrape pre-match context from FBref.com.
Falls back gracefully to historical Matches.csv for form & H2H.

Returned dict keys:
  home_form         list[str]   — up to 5 results e.g. ["W","W","D","L","W"]
  away_form         list[str]
  h2h_record        dict        — {home_wins, away_wins, draws, matches}
  home_goals_avg    float|None  — avg goals scored, last 5
  away_goals_avg    float|None
  home_conceded_avg float|None
  away_conceded_avg float|None
  source            str         — "fbref" | "historical"
"""

import time
import random
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent   # project root
MATCHES_CSV = BASE_DIR / "Data" / "Matches.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
FBREF_SEARCH = "https://fbref.com/search/search.fcgi?search={team}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _polite_get(url: str, retries: int = 3, delay: float = 2.0) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            time.sleep(delay + random.uniform(0.5, 1.5))
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                logger.warning(f"Rate-limited — waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            logger.warning(f"[Attempt {attempt+1}] {url} → {exc}")
    return None


def _result_code(home_goals: float, away_goals: float, perspective: str) -> str:
    if home_goals == away_goals:
        return "D"
    if perspective == "home":
        return "W" if home_goals > away_goals else "L"
    return "W" if away_goals > home_goals else "L"


# ── Historical CSV fallback ───────────────────────────────────────────────────

_matches_cache: Optional[pd.DataFrame] = None


def _load_matches() -> Optional[pd.DataFrame]:
    global _matches_cache
    if _matches_cache is not None:
        return _matches_cache
    try:
        df = pd.read_csv(MATCHES_CSV, parse_dates=["MatchDate"], low_memory=False)
        df = df.dropna(subset=["FTResult", "FTHome", "FTAway"])
        _matches_cache = df
        return df
    except Exception as exc:
        logger.error(f"Could not load Matches.csv: {exc}")
        return None


def _historical_form(team: str, n: int = 5) -> dict:
    df = _load_matches()
    empty = {"form": [], "goals_avg": None, "conceded_avg": None}
    if df is None:
        return empty

    mask  = (df["HomeTeam"].str.lower() == team.lower()) | \
            (df["AwayTeam"].str.lower() == team.lower())
    recent = df[mask].sort_values("MatchDate").tail(n)
    if recent.empty:
        return empty

    results, goals, conceded = [], [], []
    for _, row in recent.iterrows():
        is_home = row["HomeTeam"].lower() == team.lower()
        results.append(_result_code(row["FTHome"], row["FTAway"],
                                    "home" if is_home else "away"))
        if is_home:
            goals.append(row["FTHome"]); conceded.append(row["FTAway"])
        else:
            goals.append(row["FTAway"]); conceded.append(row["FTHome"])

    return {
        "form":          results,
        "goals_avg":     round(sum(goals)    / len(goals),    2),
        "conceded_avg":  round(sum(conceded) / len(conceded), 2),
    }


def _historical_h2h(home_team: str, away_team: str, n: int = 5) -> dict:
    df = _load_matches()
    empty = {"home_wins": 0, "away_wins": 0, "draws": 0, "matches": []}
    if df is None:
        return empty

    mask = (
        (df["HomeTeam"].str.lower() == home_team.lower()) &
        (df["AwayTeam"].str.lower() == away_team.lower())
    ) | (
        (df["HomeTeam"].str.lower() == away_team.lower()) &
        (df["AwayTeam"].str.lower() == home_team.lower())
    )
    h2h = df[mask].sort_values("MatchDate").tail(n)

    hw, aw, dr, lst = 0, 0, 0, []
    for _, row in h2h.iterrows():
        home_is_home = row["HomeTeam"].lower() == home_team.lower()
        if row["FTHome"] == row["FTAway"]:
            dr += 1
            score = f"{int(row['FTHome'])}-{int(row['FTAway'])}"
        elif row["FTHome"] > row["FTAway"]:
            (hw if home_is_home else aw).__class__  # type annotation trick
            if home_is_home: hw += 1
            else: aw += 1
            score = (f"{int(row['FTHome'])}-{int(row['FTAway'])}" if home_is_home
                     else f"{int(row['FTAway'])}-{int(row['FTHome'])}")
        else:
            if home_is_home: aw += 1
            else: hw += 1
            score = (f"{int(row['FTHome'])}-{int(row['FTAway'])}" if home_is_home
                     else f"{int(row['FTAway'])}-{int(row['FTHome'])}")
        lst.append({"date": str(row["MatchDate"].date()), "score": score})

    return {"home_wins": hw, "away_wins": aw, "draws": dr, "matches": lst}


# ── FBref scraper ─────────────────────────────────────────────────────────────

def _fbref_team_url(team: str) -> Optional[str]:
    resp = _polite_get(FBREF_SEARCH.format(team=team.replace(" ", "+")))
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    for item in soup.select(".search-item-name a"):
        href = item.get("href", "")
        if "/squads/" in href:
            return "https://fbref.com" + href
    return None


def _fbref_form(team_url: str, n: int = 5) -> dict:
    scores_url = team_url.rstrip("/") + "/matchlogs/all_comps/schedule/"
    resp = _polite_get(scores_url)
    if not resp:
        return {"form": [], "goals_avg": None, "conceded_avg": None}

    soup  = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id=lambda x: x and "matchlogs" in x.lower()) \
            or soup.find("table")
    if not table or not table.find("tbody"):
        return {"form": [], "goals_avg": None, "conceded_avg": None}

    played = []
    for row in table.find("tbody").find_all("tr"):
        r = row.find("td", {"data-stat": "result"})
        if r and r.text.strip() in ("W", "D", "L"):
            played.append(row)
    played = played[-n:]

    results, goals, conceded = [], [], []
    for row in played:
        r  = row.find("td", {"data-stat": "result"})
        gf = row.find("td", {"data-stat": "goals_for"})
        ga = row.find("td", {"data-stat": "goals_against"})
        if r:  results.append(r.text.strip())
        if gf and gf.text.strip().isdigit(): goals.append(int(gf.text.strip()))
        if ga and ga.text.strip().isdigit(): conceded.append(int(ga.text.strip()))

    return {
        "form":         results,
        "goals_avg":    round(sum(goals)    / len(goals),    2) if goals    else None,
        "conceded_avg": round(sum(conceded) / len(conceded), 2) if conceded else None,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def scrape_prematch_context(
    home_team: str,
    away_team: str,
    use_fbref: bool = True,
) -> dict:
    """
    Fetch pre-match context. Attempts FBref first; falls back to Matches.csv.
    Never raises — always returns a dict (may contain empty lists / None values).
    """
    source    = "historical"
    home_data = {"form": [], "goals_avg": None, "conceded_avg": None}
    away_data = {"form": [], "goals_avg": None, "conceded_avg": None}

    if use_fbref:
        for team, store in [(home_team, "home"), (away_team, "away")]:
            try:
                url = _fbref_team_url(team)
                if url:
                    data = _fbref_form(url)
                    if data["form"]:
                        if store == "home": home_data = data
                        else:              away_data = data
                        source = "fbref"
            except Exception as exc:
                logger.warning(f"FBref failed for {team}: {exc}")

    if not home_data["form"]:
        home_data = _historical_form(home_team)
    if not away_data["form"]:
        away_data = _historical_form(away_team)
    if home_data["form"] or away_data["form"]:
        source = source if source == "fbref" else "historical"

    h2h = _historical_h2h(home_team, away_team)

    return {
        "home_form":         home_data["form"],
        "away_form":         away_data["form"],
        "h2h_record":        h2h,
        "home_goals_avg":    home_data["goals_avg"],
        "away_goals_avg":    away_data["goals_avg"],
        "home_conceded_avg": home_data["conceded_avg"],
        "away_conceded_avg": away_data["conceded_avg"],
        "source":            source,
    }
