"""
src/scraper.py — FootballIQ Live Data Scraper
==============================================
Fetches pre-match context from the LiveScore API (via RapidAPI).
Falls back gracefully to historical Matches.csv for form & H2H.

LiveScore API flow:
  1. v2/search  → resolve team name → team ID
  2. matches/v2/list-by-date  → find a recent match Eid involving either team
  3. matches/v2/get-pregame-form  → use match Eid to get recent form (EL array)

Returned dict keys:
  home_form         list[str]   — up to 5 results e.g. ["W","W","D","L","W"]
  away_form         list[str]
  h2h_record        dict        — {home_wins, away_wins, draws, matches}
  home_goals_avg    float|None  — avg goals scored, last 5
  away_goals_avg    float|None
  home_conceded_avg float|None
  away_conceded_avg float|None
  source            str         — "livescore" | "historical"
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent   # project root
MATCHES_CSV = BASE_DIR / "Data" / "Matches.csv"

load_dotenv(BASE_DIR / ".env")

# ── LiveScore RapidAPI config ─────────────────────────────────────────────────
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "livescore6.p.rapidapi.com"

RAPIDAPI_HEADERS = {
    "x-rapidapi-key":  RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result_code(home_goals: float, away_goals: float, perspective: str) -> str:
    """Return W/D/L from perspective of 'home' or 'away'."""
    if home_goals == away_goals:
        return "D"
    if perspective == "home":
        return "W" if home_goals > away_goals else "L"
    return "W" if away_goals > home_goals else "L"


def _api_get(endpoint: str, params: dict, retries: int = 2) -> Optional[dict]:
    """Make a GET request to the LiveScore RapidAPI. Returns parsed JSON or None."""
    if not RAPIDAPI_KEY:
        logger.warning("RAPIDAPI_KEY not set — skipping LiveScore API call")
        return None

    url = f"https://{RAPIDAPI_HOST}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(
                url, headers=RAPIDAPI_HEADERS, params=params,
                timeout=15, allow_redirects=False,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                logger.warning(f"Rate-limited — waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code in (301, 302):
                logger.warning(f"{endpoint} returned redirect ({resp.status_code})")
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning(f"[Attempt {attempt+1}] {url} → {exc}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Historical CSV fallback
# ══════════════════════════════════════════════════════════════════════════════

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
    recent = df[mask].sort_values("MatchDate", ascending=False).head(n)
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


def _historical_venue_form(team: str, is_home: bool, n: int = 5) -> dict:
    df = _load_matches()
    empty = {"form": [], "goals_avg": None, "conceded_avg": None}
    if df is None:
        return empty

    if is_home:
        mask = (df["HomeTeam"].str.lower() == team.lower())
    else:
        mask = (df["AwayTeam"].str.lower() == team.lower())
        
    recent = df[mask].sort_values("MatchDate", ascending=False).head(n)
    if recent.empty:
        return empty

    results, goals, conceded = [], [], []
    for _, row in recent.iterrows():
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
        score = f"{int(row['FTHome'])}-{int(row['FTAway'])}"
        if row["FTHome"] == row["FTAway"]:
            dr += 1
        elif row["FTHome"] > row["FTAway"]:
            if home_is_home: hw += 1
            else: aw += 1
        else:
            if home_is_home: aw += 1
            else: hw += 1
        lst.append({"date": str(row["MatchDate"].date()), "score": score})

    return {"home_wins": hw, "away_wins": aw, "draws": dr, "matches": lst}


# ══════════════════════════════════════════════════════════════════════════════
#  LiveScore API integration
# ══════════════════════════════════════════════════════════════════════════════

_team_id_cache: dict = {}   # team_name_lower -> ID string


def _livescore_search_team(team: str) -> Optional[str]:
    """Search LiveScore for a team and return its ID string.

    Response: {"Teams": [{"ID":"2773","Nm":"Arsenal","CoNm":"England",...}]}
    """
    key = team.lower()
    if key in _team_id_cache:
        return _team_id_cache[key]

    data = _api_get("v2/search", {"Category": "soccer", "Query": team})
    if not data or not isinstance(data, dict):
        return None

    try:
        teams_list = data.get("Teams", [])
        if not teams_list:
            logger.warning(f"LiveScore: no teams for '{team}'. Keys: {list(data.keys())}")
            return None

        team_lower = team.lower()

        # Handle common CSV names -> LiveScore names
        aliases = {
            "newcastle": "newcastle united",
            "wolves": "wolverhampton wanderers",
            "spurs": "tottenham hotspur",
            "man united": "manchester united",
            "man utd": "manchester united",
            "man city": "manchester city",
            "sheff utd": "sheffield united",
            "nott'm forest": "nottingham forest",
        }
        search_target = aliases.get(team_lower, team_lower)

        # Pass 1: exact name match
        for t in teams_list:
            if t.get("Nm", "").lower() == search_target:
                tid = str(t["ID"])
                _team_id_cache[key] = tid
                logger.info(f"LiveScore: '{team}' → ID={tid} (exact)")
                return tid

        # Pass 2: partial match, prefer major leagues, ignore youth/women
        major = {"England", "Spain", "Germany", "Italy", "France"}
        ignore = {" u21", " u19", " u18", " w", " women", " reserve", " youth"}
        
        for t in teams_list:
            nm = t.get("Nm", "")
            nm_lower = nm.lower()
            
            if any(x in nm_lower for x in ignore):
                continue
                
            if search_target in nm_lower and t.get("CoNm") in major:
                tid = str(t["ID"])
                _team_id_cache[key] = tid
                logger.info(f"LiveScore: '{team}' → ID={tid} (name='{nm}')")
                return tid

        # Pass 3: first result as fallback (also ignoring youth/women if possible)
        for t in teams_list:
            nm_lower = t.get("Nm", "").lower()
            if not any(x in nm_lower for x in ignore):
                tid = str(t["ID"])
                _team_id_cache[key] = tid
                logger.info(f"LiveScore: '{team}' → ID={tid} (fallback: '{t.get('Nm')}')")
                return tid
                
        # If absolutely everything is youth/women (unlikely), just take the first
        first = teams_list[0]
        tid = str(first["ID"])
        _team_id_cache[key] = tid
        logger.info(f"LiveScore: '{team}' → ID={tid} (desperate fallback: '{first.get('Nm')}')")
        return tid

    except Exception as exc:
        logger.warning(f"Error parsing LiveScore search for '{team}': {exc}")
    return None


def _find_match_eid_for_team(team_id: str, days_back: int = 7) -> Optional[str]:
    """Scan recent dates to find ANY match involving the team. Returns match Eid."""
    today = datetime.utcnow().date()
    for offset in range(days_back):
        dt = today - timedelta(days=offset)
        date_str = dt.strftime("%Y%m%d")

        data = _api_get("matches/v2/list-by-date",
                        {"Category": "soccer", "Date": date_str})
        if not data or not isinstance(data, dict):
            continue

        for stage in data.get("Stages", []):
            for evt in stage.get("Events", []):
                # Check if team is T1 or T2
                for side in ("T1", "T2"):
                    team_list = evt.get(side, [])
                    if isinstance(team_list, list):
                        for t in team_list:
                            if str(t.get("ID", "")) == team_id:
                                eid = str(evt["Eid"])
                                logger.info(
                                    f"LiveScore: found match Eid={eid} for team "
                                    f"ID={team_id} on {dt}"
                                )
                                return eid
    return None


def _parse_form_from_el(el_events: list, team_id: str, n: int = 5) -> dict:
    """
    Parse the EL (Event List) from get-pregame-form into W/D/L form.

    Each EL event has:
      T1: [{"ID":"2772","Nm":"Crystal Palace"}]
      T2: [{"ID":"252","Nm":"West Ham United"}]
      Tr1: "0"   (home goals)
      Tr2: "0"   (away goals)
      Eps: "FT"  (finished)
    """
    empty = {"form": [], "goals_avg": None, "conceded_avg": None}
    results, goals_scored, goals_conceded = [], [], []

    for evt in el_events[:n]:
        # Get scores
        try:
            tr1 = int(evt.get("Tr1", -1))
            tr2 = int(evt.get("Tr2", -1))
        except (ValueError, TypeError):
            continue
        if tr1 < 0 or tr2 < 0:
            continue

        # Only count finished matches
        eps = evt.get("Eps", "")
        if isinstance(eps, str) and eps not in ("FT", "AET", "AP"):
            continue

        # Determine if our team is T1 (home) or T2 (away)
        t1_list = evt.get("T1", [])
        t1_id = str(t1_list[0].get("ID", "")) if isinstance(t1_list, list) and t1_list else ""

        is_home = (t1_id == team_id)

        perspective = "home" if is_home else "away"
        results.append(_result_code(tr1, tr2, perspective))

        gf = tr1 if is_home else tr2
        ga = tr2 if is_home else tr1
        goals_scored.append(gf)
        goals_conceded.append(ga)

    if not results:
        return empty

    return {
        "form":         results,
        "goals_avg":    round(sum(goals_scored)   / len(goals_scored),   2),
        "conceded_avg": round(sum(goals_conceded) / len(goals_conceded), 2),
    }


def _parse_venue_form_from_el(el_events: list, team_id: str, is_home_venue: bool, n: int = 5) -> dict:
    empty = {"form": [], "goals_avg": None, "conceded_avg": None}
    results, goals_scored, goals_conceded = [], [], []

    for evt in el_events:
        # Get scores
        try:
            tr1 = int(evt.get("Tr1", -1))
            tr2 = int(evt.get("Tr2", -1))
        except (ValueError, TypeError):
            continue
        if tr1 < 0 or tr2 < 0:
            continue

        # Only count finished matches
        eps = evt.get("Eps", "")
        if isinstance(eps, str) and eps not in ("FT", "AET", "AP"):
            continue

        # Determine if our team is T1 (home) or T2 (away)
        t1_list = evt.get("T1", [])
        t1_id = str(t1_list[0].get("ID", "")) if isinstance(t1_list, list) and t1_list else ""
        is_home = (t1_id == team_id)

        # Filter by venue
        if is_home != is_home_venue:
            continue

        perspective = "home" if is_home else "away"
        results.append(_result_code(tr1, tr2, perspective))

        gf = tr1 if is_home else tr2
        ga = tr2 if is_home else tr1
        goals_scored.append(gf)
        goals_conceded.append(ga)

        if len(results) >= n:
            break

    if not results:
        return empty

    return {
        "form":         results,
        "goals_avg":    round(sum(goals_scored)   / len(goals_scored),   2),
        "conceded_avg": round(sum(goals_conceded) / len(goals_conceded), 2),
    }


def _livescore_form_via_pregame(
    home_team_id: str,
    away_team_id: str,
    n: int = 5,
) -> tuple[dict, dict, dict, dict]:
    """
    Try to get form for both teams using matches/v2/get-pregame-form.

    Returns (home_data, away_data, home_venue_data, away_venue_data).
    """
    empty = {"form": [], "goals_avg": None, "conceded_avg": None}

    # Step 1: find a match Eid for either team
    match_eid = _find_match_eid_for_team(home_team_id, days_back=4)
    if not match_eid:
        match_eid = _find_match_eid_for_team(away_team_id, days_back=4)
    if not match_eid:
        logger.warning("LiveScore: no recent match found for either team")
        return empty, empty, empty, empty

    # Step 2: get pregame form
    data = _api_get("matches/v2/get-pregame-form", {
        "Category": "soccer",
        "Eid": match_eid,
    })
    if not data or not isinstance(data, dict):
        return empty, empty

    # Step 3: parse T1 and T2 form data
    home_data, away_data = empty.copy(), empty.copy()
    home_venue_data, away_venue_data = empty.copy(), empty.copy()

    for side_key in ("T1", "T2"):
        side_list = data.get(side_key, [])
        if not isinstance(side_list, list) or not side_list:
            continue

        team_obj = side_list[0]
        tid = str(team_obj.get("ID", ""))
        el_events = team_obj.get("EL", [])

        if not el_events:
            continue

        form_data = _parse_form_from_el(el_events, tid, n)

        if tid == home_team_id:
            home_data = form_data
            home_venue_data = _parse_venue_form_from_el(el_events, tid, True, n)
        elif tid == away_team_id:
            away_data = form_data
            away_venue_data = _parse_venue_form_from_el(el_events, tid, False, n)

    # If we only found form for one team (the other wasn't in this match),
    # try finding a separate match for the missing team
    if not home_data["form"] and home_team_id:
        home_data, home_venue_data = _fallback_form_via_dates(home_team_id, True, n)
    if not away_data["form"] and away_team_id:
        away_data, away_venue_data = _fallback_form_via_dates(away_team_id, False, n)

    return home_data, away_data, home_venue_data, away_venue_data


def _fallback_form_via_dates(team_id: str, is_home_venue: bool, n: int = 5) -> tuple[dict, dict]:
    """
    Fallback: find a match for this specific team and use get-pregame-form.
    Returns (overall_form, venue_form).
    """
    empty = {"form": [], "goals_avg": None, "conceded_avg": None}

    match_eid = _find_match_eid_for_team(team_id, days_back=7)
    if not match_eid:
        return empty, empty

    data = _api_get("matches/v2/get-pregame-form", {
        "Category": "soccer",
        "Eid": match_eid,
    })
    if not data or not isinstance(data, dict):
        return empty, empty

    for side_key in ("T1", "T2"):
        side_list = data.get(side_key, [])
        if not isinstance(side_list, list) or not side_list:
            continue
        team_obj = side_list[0]
        tid = str(team_obj.get("ID", ""))
        if tid == team_id:
            el_events = team_obj.get("EL", [])
            if el_events:
                return _parse_form_from_el(el_events, tid, n), _parse_venue_form_from_el(el_events, tid, is_home_venue, n)

    return empty, empty


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def scrape_prematch_context(
    home_team: str,
    away_team: str,
    use_live_data: bool = True,
) -> dict:
    """
    Fetch pre-match context. Attempts LiveScore API first; falls back to CSV.
    Never raises — always returns a dict (may contain empty lists / None values).
    """
    source    = "historical"
    home_data = {"form": [], "goals_avg": None, "conceded_avg": None}
    away_data = {"form": [], "goals_avg": None, "conceded_avg": None}

    home_venue_form = []
    away_venue_form = []

    if use_live_data and RAPIDAPI_KEY:
        try:
            # Step 1: resolve team IDs
            home_id = _livescore_search_team(home_team)
            away_id = _livescore_search_team(away_team)

            if home_id or away_id:
                # Step 2+3: get form via pregame endpoint
                hd, ad, hvd, avd = _livescore_form_via_pregame(
                    home_id or "", away_id or "", n=5,
                )
                if hd["form"]:
                    home_data = hd
                    home_venue_form = hvd.get("form", [])
                    source = "livescore"
                if ad["form"]:
                    away_data = ad
                    away_venue_form = avd.get("form", [])
                    source = "livescore"

        except Exception as exc:
            logger.warning(f"LiveScore pipeline failed: {exc}")

    elif use_live_data and not RAPIDAPI_KEY:
        logger.warning("Live data requested but RAPIDAPI_KEY not set in .env")

    # Fallback to historical CSV for any missing data
    if not home_data["form"]:
        home_data = _historical_form(home_team)
    if not away_data["form"]:
        away_data = _historical_form(away_team)

    h2h = _historical_h2h(home_team, away_team)

    return {
        "home_form":         home_data["form"],
        "away_form":         away_data["form"],
        "home_team_home_form": home_venue_form,
        "away_team_away_form": away_venue_form,
        "h2h_record":        h2h,
        "home_goals_avg":    home_data["goals_avg"],
        "away_goals_avg":    away_data["goals_avg"],
        "home_conceded_avg": home_data["conceded_avg"],
        "away_conceded_avg": away_data["conceded_avg"],
        "source":            source,
    }
