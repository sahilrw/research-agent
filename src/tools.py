"""Web search and fetch tools.

Exposes:
- tavily_search(query, ...) -> list[SearchHit]
- fetch_page(url) -> PageContent   (only if Tavily extracts are insufficient)
"""

import os
import json
import hashlib
import logging
from urllib.parse import urlparse
from typing import List
from tavily import TavilyClient

from src.report import Source

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

CACHE_DIR = ".cache/tavily"

def clean_domain(url: str) -> str:
    """Extracts netlock from url and strips 'www.'"""
    try:
        netloc = urlparse(url).netloc
        if netloc.startswith("www."):
            return netloc[4:]
        return netloc
    except Exception:
        return "Unknown"
    
def tavily_search(query: str, max_results: int = 5, search_depth: str = "basic") -> List[Source]:
    """
    Executes a web search via Tavily. Utilizes a deterministic JSON local filesystem 
    cache to bypass hitting the live API for identical query setups.
    """
    if not tavily_client:
        print("Error: Tavily client is not initialized. Check TAVILY_API_KEY.")
        return []

    normalized_q = " ".join(query.lower().split())
    payload = json.dumps({"q": normalized_q, "depth": search_depth, "k": max_results}, sort_keys=True)
    key = hashlib.sha256(payload.encode()).hexdigest()[:16]

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{key}.json")

    if os.path.exists(cache_path):
        try: 
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return [Source.model_validate(item) for item in cached_data]
        except Exception as e:
            print(f"Cache corrupt for path {cache_path}: {e}. Falling back to live API.")
    
    try: 
        response = tavily_client.search(query=query, search_depth=search_depth, max_results=max_results)
        raw_results = response.get("results", [])
    except Exception as e:
        print(f"Tavily live search failed: {e}")
        return []
    
    sources = []
    for res in raw_results:
        sources.append(Source(
            url = res["url"],
            title = res["title"], 
            domain = clean_domain(res["url"]),
            snippet=res.get("content", ""),
            published=res.get("published_date"),
        ))

    try:
        serialized_payload = [source.model_dump() for source in sources]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(serialized_payload, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to write cache entry to {cache_path}: {e}")
    
    return sources