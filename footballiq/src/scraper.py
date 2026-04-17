"""
src/scraper.py — FootballIQ Live Data Scraper
==============================================
Attempts to scrape pre-match context from FBref.com.
Falls back to historical Matches.csv for H2H and form if scraping fails.

Output dict keys:
  home_form        : list of up-to-5 result strings e.g. ["W","W","D","L","W"]
  away_form        : same for away team
  h2h_record       : {"home_wins":int, "away_wins":int, "draws":int, "matches":list}
  home_goals_avg   : float — avg goals scored last 5
  away_goals_avg   : float
  home_conceded_avg: float
  away_conceded_avg: float
  source           : "fbref" | "historical" | "simulated"
"""

import time
import random
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent          # footballiq/
DATA_DIR    = BASE_DIR / "data"
MATCHES_CSV = DATA_DIR / "Matches.csv"
if not MATCHES_CSV.exists():
    MATCHES_CSV = BASE_DIR.parent / "Data" / "Matches.csv"   # project root fallback

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
    """GET with retry + random jitter to respect rate limits."""
    for attempt in range(retries):
        try:
            time.sleep(delay + random.uniform(0.5, 1.5))
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                logger.warning(f"Rate-limited. Waiting {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            logger.warning(f"[Attempt {attempt+1}] GET {url} failed: {exc}")
    return None


def _result_code(home_goals: int, away_goals: int, perspective: str) -> str:
    """Return W/D/L from a team's perspective."""
    if home_goals == away_goals:
        return "D"
    if perspective == "home":
        return "W" if home_goals > away_goals else "L"
    return "W" if away_goals > home_goals else "L"


# ── Historical fallback (from Matches.csv) ───────────────────────────────────

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
    """Pull last-n results + goals from Matches.csv."""
    df = _load_matches()
    default = {"form": [], "goals_avg": None, "conceded_avg": None}
    if df is None:
        return default

    home_mask = df["HomeTeam"].str.lower() == team.lower()
    away_mask = df["AwayTeam"].str.lower() == team.lower()
    team_matches = df[home_mask | away_mask].sort_values("MatchDate").tail(n)

    if team_matches.empty:
        return default

    results, goals, conceded = [], [], []
    for _, row in team_matches.iterrows():
        if row["HomeTeam"].lower() == team.lower():
            results.append(_result_code(row["FTHome"], row["FTAway"], "home"))
            goals.append(row["FTHome"])
            conceded.append(row["FTAway"])
        else:
            results.append(_result_code(row["FTHome"], row["FTAway"], "away"))
            goals.append(row["FTAway"])
            conceded.append(row["FTHome"])

    return {
        "form":          results,
        "goals_avg":     round(sum(goals) / len(goals), 2) if goals else None,
        "conceded_avg":  round(sum(conceded) / len(conceded), 2) if conceded else None,
    }


def _historical_h2h(home_team: str, away_team: str, n: int = 5) -> dict:
    """Pull last-n head-to-head results from Matches.csv."""
    df = _load_matches()
    default = {"home_wins": 0, "away_wins": 0, "draws": 0, "matches": []}
    if df is None:
        return default

    mask = (
        (df["HomeTeam"].str.lower() == home_team.lower()) & (df["AwayTeam"].str.lower() == away_team.lower())
    ) | (
        (df["HomeTeam"].str.lower() == away_team.lower()) & (df["AwayTeam"].str.lower() == home_team.lower())
    )
    h2h = df[mask].sort_values("MatchDate").tail(n)

    hw, aw, dr = 0, 0, 0
    matches_list = []
    for _, row in h2h.iterrows():
        home_won = row["FTHome"] > row["FTAway"]
        away_won = row["FTAway"] > row["FTHome"]
        draw     = row["FTHome"] == row["FTAway"]

        # Normalise perspective: home_team is always "home" side
        if row["HomeTeam"].lower() == home_team.lower():
            if home_won: hw += 1
            elif away_won: aw += 1
            else: dr += 1
            score = f"{int(row['FTHome'])}-{int(row['FTAway'])}"
        else:
            # teams reversed
            if home_won: aw += 1
            elif away_won: hw += 1
            else: dr += 1
            score = f"{int(row['FTAway'])}-{int(row['FTHome'])}"

        matches_list.append({
            "date":  str(row["MatchDate"].date()),
            "score": score,
        })

    return {"home_wins": hw, "away_wins": aw, "draws": dr, "matches": matches_list}


# ── FBref scraper ─────────────────────────────────────────────────────────────

def _fbref_team_url(team: str) -> Optional[str]:
    """Find the FBref team page URL via the search endpoint."""
    resp = _polite_get(FBREF_SEARCH.format(team=team.replace(" ", "+")))
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    # Search results are in <div class="search-item-name"> → first <a>
    for item in soup.select(".search-item-name a"):
        href = item.get("href", "")
        if "/squads/" in href:
            return "https://fbref.com" + href
    return None


def _fbref_form(team_url: str, team_name: str, n: int = 5) -> dict:
    """Scrape the last-n match results from a FBref team's scores & fixtures page."""
    scores_url = team_url.rstrip("/") + "/matchlogs/all_comps/schedule/"
    resp = _polite_get(scores_url)
    if not resp:
        return {"form": [], "goals_avg": None, "conceded_avg": None}

    soup  = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id=lambda x: x and "matchlogs" in x.lower())
    if not table:
        # Try selector fallback
        table = soup.find("table")

    if not table:
        return {"form": [], "goals_avg": None, "conceded_avg": None}

    results, goals, conceded = [], [], []
    rows = table.find("tbody").find_all("tr") if table.find("tbody") else []
    # Filter to rows with a result (played matches) and take last n
    played = []
    for row in rows:
        result_cell = row.find("td", {"data-stat": "result"})
        if result_cell and result_cell.text.strip() in ("W", "D", "L"):
            played.append(row)
    played = played[-n:]

    for row in played:
        result_cell  = row.find("td", {"data-stat": "result"})
        gf_cell      = row.find("td", {"data-stat": "goals_for"})
        ga_cell      = row.find("td", {"data-stat": "goals_against"})
        if result_cell:
            results.append(result_cell.text.strip())
        if gf_cell and gf_cell.text.strip().isdigit():
            goals.append(int(gf_cell.text.strip()))
        if ga_cell and ga_cell.text.strip().isdigit():
            conceded.append(int(ga_cell.text.strip()))

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
    Fetch pre-match context for a fixture.

    Parameters
    ----------
    home_team   : Display name of the home team.
    away_team   : Display name of the away team.
    use_fbref   : Attempt FBref scrape first; fall back to historical CSV.

    Returns
    -------
    dict with keys: home_form, away_form, h2h_record,
                    home_goals_avg, away_goals_avg,
                    home_conceded_avg, away_conceded_avg, source
    """
    source = "historical"
    home_data = {"form": [], "goals_avg": None, "conceded_avg": None}
    away_data = {"form": [], "goals_avg": None, "conceded_avg": None}

    # ── Try FBref ─────────────────────────────────────────────────────────
    if use_fbref:
        try:
            home_url = _fbref_team_url(home_team)
            if home_url:
                home_fbref = _fbref_form(home_url, home_team)
                if home_fbref["form"]:
                    home_data = home_fbref
                    source = "fbref"
        except Exception as exc:
            logger.warning(f"FBref scrape failed for {home_team}: {exc}")

        try:
            away_url = _fbref_team_url(away_team)
            if away_url:
                away_fbref = _fbref_form(away_url, away_team)
                if away_fbref["form"]:
                    away_data = away_fbref
                    source = "fbref"
        except Exception as exc:
            logger.warning(f"FBref scrape failed for {away_team}: {exc}")

    # ── Fallback to historical CSV ────────────────────────────────────────
    if not home_data["form"]:
        home_data = _historical_form(home_team)
        if home_data["form"]:
            source = "historical"

    if not away_data["form"]:
        away_data = _historical_form(away_team)
        if away_data["form"]:
            source = "historical"

    # ── H2H always from historical CSV (most reliable) ───────────────────
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
