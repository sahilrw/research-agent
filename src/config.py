# src/config.py

# The Curated Tier Registry (35 high-impact reference domains)
DOMAIN_TIERS: dict[str, int] = {
    # Tier 1: Peer-reviewed / Authoritative Entities (1.00)
    "nature.com": 1,
    "science.org": 1,
    "pubmed.ncbi.nlm.nih.gov": 1,
    "ipcc.ch": 1,
    "who.int": 1,
    "ec.europa.eu": 1,
    
    # Tier 2: Major News with Stringent Editorial Pipelines (0.80)
    "reuters.com": 2,
    "apnews.com": 2,
    "bbc.com": 2,
    "ft.com": 2,
    "nytimes.com": 2,
    "economist.com": 2,
    "theguardian.com": 2,
    "bloomberg.com": 2,
    
    # Tier 3: Established Corporate Research & Reference Libraries (0.60)
    "brookings.edu": 3,
    "rand.org": 3,
    "mckinsey.com": 3,
    "wikipedia.org": 3,
    "arxiv.org": 3,
    "redis.io": 3,
    "docs.python.org": 3,
    "microsoft.com": 3,
    
    # Tier 4: Public Technical Registries & Content Aggregators (0.40)
    "substack.com": 4,
    "medium.com": 4,
    "stackoverflow.com": 4,
    "towardsdatascience.com": 4,
    "github.com": 4,
    "dev.to": 4,
    "reddit.com": 4
}

# Tier Scalar Mappings
TIER_SCORES: dict[int, float] = {
    1: 1.00,
    2: 0.80,
    3: 0.60,
    4: 0.40,
    5: 0.20  # Fallback allocation baseline for unknown vectors
}

DEFAULT_TIER: int = 5

# Connective Tissue: Planner Strategy -> Temporal Decay Rate
HALF_LIFE_DAYS: dict[bool, int] = {
    True: 90,     # tight temporal horizon (3 months)
    False: 365,   # relaxed historical horizon (1 year)
}

# Defensive Weight Matrix Configuration
WEIGHTS: dict[str, float] = {
    "domain": 0.7,
    "recency": 0.3
}

MISSING_DATE_SCORE: float = 0.5

# Threshold baseline below which a node must spawn deep-dive follow-up research
CONFIDENCE_THRESHOLD: float = 0.7

# Maximum depth layers the agent can venture down to prevent infinite recursion trees
MAX_DEPTH: int = 3

# Upper bound restriction on children nodes spawned by any single gap critique phase
MAX_FOLLOW_UPS: int = 2

# Global budget cap for total sub-questions allowed across the entire run
MAX_TOTAL_SUB_QUESTIONS: int = 10