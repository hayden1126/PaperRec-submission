# PaperRec: Academic Paper Recommendation via Personalized PageRank

**Course:** MATH UN2015 - Linear Algebra & Probability, Columbia University
**Authors:** Hayden Leung (cl4627), Rea Lila (rl3512), Mireille Donoso-Kugler (md4418)

## Where to look

* **Paper:** [`paper/paper.pdf`](paper/paper.pdf)
* **Canonical run** (the figures and rankings used in the paper, scraped 2026-04-25): [`results/`](results/)
  - `edges.csv` — scraped citation network (3,158 edges, 2,504 nodes)
  - `ranked_output.csv` — top-20 papers by Personalized PageRank
  - `fig1_convergence.pdf`, `fig2_damping.pdf` — paper figures
  - `seed.txt` — resolved Semantic Scholar ID of the seed (*Attention Is All You Need*)
* **Code:** Python modules at the repository root.

## What this is

A recommendation engine for the cold-start literature-review problem. Given one seed paper, the pipeline scrapes a citation ego-network via BFS through the Semantic Scholar API, models the network as a Markov chain, and ranks neighbours by Personalized PageRank. The paper analyzes the algorithm's convergence (Method A) and damping sensitivity (Method B) on a real ego-network.

## Reproducing

### Prerequisites
* Python 3.12+
* Optional: `S2_API_KEY` environment variable to lift the public-API rate limit

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the recommender
```bash
# Default seed (set in config.py: ARXIV:1706.03762, "Attention Is All You Need")
python main.py

# Custom seed
python main.py --seed <semantic_scholar_id_or_external_id>

# Adjust BFS parameters
python main.py --seed <id> --max-depth 3 --max-branching 30
```

Each run writes `edges.csv`, `ranked_output.csv`, and `seed.txt` to a fresh `output/<timestamp>/` directory.

### Regenerate the paper figures
```bash
python make_figures.py --run-dir results
```

This rewrites `fig1_convergence.pdf` and `fig2_damping.pdf` in the target directory and prints the top-20 overlap between PPR and raw citation count.

## Modules

* `main.py` — entry point; BFS scraper, metadata fetch, output formatting.
* `math_engine.py` — sparse adjacency, column-stochastic transition matrix, PPR linear solve.
* `evaluation.py` — power iteration with residual trace, damping sweep with pairwise Pearson correlation.
* `make_figures.py` — generates the two PDF figures and prints the citation-count overlap.
* `config.py` — seed paper ID, BFS depth/branching, damping factor, rate-limit constants.
