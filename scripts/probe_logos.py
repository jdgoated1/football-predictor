"""Inspect the specific league directories for missing teams."""
import requests

for league in ["Spain - LaLiga", "Germany - Bundesliga"]:
    url = f"https://api.github.com/repos/luukhopman/football-logos/contents/logos/{league}"
    r = requests.get(url)
    print(f"\n{league}:")
    for f in r.json():
        if f.get("name", "").lower().endswith(".png"):
            print(f"  {f['name']}")
