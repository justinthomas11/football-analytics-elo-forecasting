import requests, json, os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RAPIDAPI_KEY", "")
HOST = "livescore6.p.rapidapi.com"
HEADERS = {"x-rapidapi-key": API_KEY, "x-rapidapi-host": HOST}

print("Searching for 'Newcastle'...")
resp = requests.get(
    f"https://{HOST}/v2/search",
    headers=HEADERS,
    params={"Category": "soccer", "Query": "Newcastle"}
)
data = resp.json()
teams = data.get("Teams", [])
for i, t in enumerate(teams[:5]):
    print(f"{i+1}. ID: {t.get('ID')}, Name: {t.get('Nm')}, Country: {t.get('CoNm')}")
