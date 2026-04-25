"""Shared constants for the PaperRec pipeline."""

import os

# Pipeline parameters
SEED_ID = "ARXIV:1706.03762" # Attention Is All You Need
MAX_DEPTH = 2
MAX_BRANCHING = 50
DAMPING = 0.85
TOP_N = 20

# Semantic Scholar
S2_API_KEY = os.environ.get("S2_API_KEY")
REQUEST_SLEEP_SECONDS = 0.1 if S2_API_KEY else 1.1
REQUEST_TIMEOUT_SECONDS = 30

# Files
OUTPUT_DIR = "output"
EDGES_CSV = "edges.csv"
RANKED_CSV = "ranked_output.csv"
