"""
Pulls recent posts from specific X accounts via Tavily search, for TONE reference.
Runs standalone, NOT part of the main graph — data refresh job, run manually or weekly.

Caveat: Tavily indexes whatever Google/Bing have indexed of X — results can be
partial (snippets, not full tweet text) and won't cover very recent posts.
Good enough for structural/tone patterns, not a complete archive.
"""

import json
import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

ACCOUNTS = ["adxtyahq", "Av1dlive"]

OUTPUT_FILE = "structure_cache.json"


def fetch_account_posts(username: str, max_results: int = 10) -> list[dict]:
    # Tavily rejects site:-only queries — need real search terms too
    query = f"site:x.com/{username} tweets OR posts from @{username}"

    results = tavily_client.search(
        query=query,
        max_results=max_results,
    )

    posts = []
    for r in results.get("results", []):
        posts.append({
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),  # snippet text, often partial
        })

    return posts


def refresh_all():
    cache = {}

    for account in ACCOUNTS:
        print(f"Fetching posts for @{account}...")
        posts = fetch_account_posts(account)
        cache[account] = posts
        print(f"  got {len(posts)} results")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    refresh_all()
