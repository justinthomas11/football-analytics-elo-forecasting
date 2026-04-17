import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.scraper import _sofascore_team_id, _sofascore_form, scrape_prematch_context

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    t1 = "Chelsea"
    t2 = "Arsenal"
    
    print(f"Testing {t1}")
    id1 = _sofascore_team_id(t1)
    print(f"ID for {t1}: {id1}")
    if id1:
        form1 = _sofascore_form(id1)
        print(f"Form for {t1}: {form1}")
        
    print(f"Testing {t2}")
    id2 = _sofascore_team_id(t2)
    print(f"ID for {t2}: {id2}")
    if id2:
        form2 = _sofascore_form(id2)
        print(f"Form for {t2}: {form2}")
        
    print("Testing scrape_prematch_context")
    res = scrape_prematch_context(t1, t2, use_live_data=True)
    print("Result:")
    print(res)
