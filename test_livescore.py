"""Phase 3: Test get-pregame-form with a MATCH Eid, and scan dates for team matches."""
import requests, json, os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RAPIDAPI_KEY", "")
HOST = "livescore6.p.rapidapi.com"
HEADERS = {"x-rapidapi-key": API_KEY, "x-rapidapi-host": HOST}

def test(label, endpoint, params):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    try:
        resp = requests.get(f"https://{HOST}/{endpoint}", headers=HEADERS,
                            params=params, timeout=15, allow_redirects=False)
        print(f"Status: {resp.status_code}")
        if resp.status_code in (301, 302):
            print(f"Redirect → {resp.headers.get('Location', 'unknown')}"); return
        if resp.status_code == 404:
            print(f"Not found"); return
        data = resp.json()
        print(json.dumps(data, indent=2, default=str)[:4000])
    except Exception as e:
        print(f"Error: {e}")

# Test 1: get-pregame-form with a MATCH Eid (Crystal Palace vs West Ham from today)
test("get-pregame-form with MATCH Eid 1529134",
     "matches/v2/get-pregame-form",
     {"Category": "soccer", "Eid": "1529134"})

# Test 2: Scan yesterday for more matches
test("list-by-date yesterday (20260420)",
     "matches/v2/list-by-date",
     {"Category": "soccer", "Date": "20260420"})

print(f"\n{'='*60}\nDONE\n{'='*60}")
