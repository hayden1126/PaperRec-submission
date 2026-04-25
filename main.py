"""PaperRec: 
scrape a citation ego-network from a seed paper, 
rank via Personalized PageRank,
and display a reading list."""

import argparse
import csv
import os
import time
from collections import deque
from collections.abc import Iterator
from datetime import datetime

from semanticscholar import SemanticScholar
from semanticscholar.SemanticScholarException import (
    ObjectNotFoundException,
    SemanticScholarException,
)

import config
import math_engine


Edge = tuple[str, str]

# S2 get_papers batches up to 500 ids per call, so TOP_N must stay under that cap.
METADATA_FIELDS = ["title", "authors", "year", "abstract", "citationCount", "venue"]


class SemanticScholarClient:
    """SemanticScholar wrapper with safe pagination and pacing."""

    def __init__(self, *, branching: int, sleep_seconds: float, timeout: float) -> None:
        self._sch = SemanticScholar(api_key=config.S2_API_KEY, timeout=timeout)
        self._branching = branching
        self._sleep = sleep_seconds

    def references(self, paper_id: str) -> list[str]:
        return self._get_neighbors(self._sch.get_paper_references, paper_id)

    def citations(self, paper_id: str) -> list[str]:
        return self._get_neighbors(self._sch.get_paper_citations, paper_id)

    def resolve_id(self, paper_id: str) -> str:
        """Resolve an external ID (e.g. ARXIV:...) to an internal paperId."""
        paper = self._sch.get_paper(paper_id, fields=["paperId"])
        if paper and paper.paperId:
            return paper.paperId
        return paper_id

    def get_papers(self, paper_ids: list[str]) -> dict:
        papers = self._sch.get_papers(paper_ids, fields=METADATA_FIELDS)
        return {p.paperId: p for p in papers if p and p.paperId}

    def _get_neighbors(self, call, paper_id: str) -> list[str]:
        # TypeError from the library = S2 returned {"data": null} for a paper whose
        # references/citations were elided by the publisher. Deterministic, no retry.
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                page = call(paper_id, limit=self._branching)
                break
            except ObjectNotFoundException:
                return []
            except TypeError:
                print(f"  [info] {paper_id}: neighbours elided by publisher, skipping", flush=True)
                time.sleep(self._sleep)
                return []
            except SemanticScholarException as exc:
                if attempt == max_attempts - 1:
                    print(f"  [warn] giving up on {paper_id} after {max_attempts} attempts: {exc}", flush=True)
                    time.sleep(self._sleep)
                    return []
                backoff = self._sleep * (2 ** attempt)
                print(f"  [warn] transient error for {paper_id} (attempt {attempt + 1}/{max_attempts}): {exc}; retrying in {backoff:.1f}s", flush=True)
                time.sleep(backoff)
        ids: list[str] = []
        for item in page[: self._branching]:
            related = item.paper
            if related and related.paperId:
                ids.append(related.paperId)
        time.sleep(self._sleep)
        return ids


def fetch_outgoing(client: SemanticScholarClient, paper_id: str) -> Iterator[Edge]:
    for neighbour in client.references(paper_id):
        yield (paper_id, neighbour)


def fetch_incoming(client: SemanticScholarClient, paper_id: str) -> Iterator[Edge]:
    for neighbour in client.citations(paper_id):
        yield (neighbour, paper_id)


def scrape(client: SemanticScholarClient, seed_id: str, max_depth: int) -> Iterator[Edge]:
    visited: set[str] = {seed_id}
    queue: deque[tuple[str, int]] = deque([(seed_id, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        print(f"[depth {depth}] fetching {current} (queue={len(queue)})", flush=True)

        for edge in fetch_outgoing(client, current):
            yield edge
            _enqueue(edge[1], depth + 1, visited, queue)

        for edge in fetch_incoming(client, current):
            yield edge
            _enqueue(edge[0], depth + 1, visited, queue)


def _enqueue(
    neighbor: str,
    depth: int,
    visited: set[str],
    queue: deque[tuple[str, int]],
) -> None:
    if neighbor not in visited:
        visited.add(neighbor)
        queue.append((neighbor, depth))



# Output / Display

def load_rankings(path: str) -> list[tuple[int, str, float]]:
    rows: list[tuple[int, str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["Rank"]), r["PaperID"], float(r["Score"])))
    return rows


def format_entry(rank: int, score: float, paper) -> str:
    if not paper:
        return f"#{rank:2d}  [score={score:.4f}]  <metadata unavailable>"
    title = paper.title or "(untitled)"
    year = paper.year or "----"
    venue = paper.venue or ""
    authors = paper.authors or []
    first_author = authors[0].name if authors else "Unknown"
    extra_authors = f" et al. ({len(authors)})" if len(authors) > 1 else ""
    abstract = (paper.abstract or "").strip().replace("\n", " ")
    if len(abstract) > 280:
        abstract = abstract[:277] + "..."
    header = f"#{rank:2d}  [score={score:.4f}]  ({year})  {title}"
    byline = f"      {first_author}{extra_authors}"
    if venue:
        byline += f"  -  {venue}"
    body = f"      {abstract}" if abstract else ""
    return "\n".join(filter(None, [header, byline, body]))


# Main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaperRec: citation-based paper recommendations.")
    p.add_argument("--seed", default=config.SEED_ID, help="Semantic Scholar paper ID")
    p.add_argument("--max-depth", type=int, default=config.MAX_DEPTH)
    p.add_argument("--max-branching", type=int, default=config.MAX_BRANCHING)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not config.S2_API_KEY:
        print("[warn] S2_API_KEY not set; API requests throttled to ~1/s. "
              "Depth-2 scrapes with branching=50 can take many minutes.", flush=True)
    client = SemanticScholarClient(
        branching=args.max_branching,
        sleep_seconds=config.REQUEST_SLEEP_SECONDS,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )

    seed_id = client.resolve_id(args.seed)
    print(f"Resolved seed: {args.seed} -> {seed_id}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(config.OUTPUT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Seed: {seed_id}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)

    edges_path = os.path.join(run_dir, config.EDGES_CSV)
    ranked_path = os.path.join(run_dir, config.RANKED_CSV)

    # 1. Scrape
    count = 0
    with open(edges_path, "w", newline="", encoding="utf-8", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target"])
        for edge in scrape(client, seed_id, args.max_depth):
            writer.writerow(edge)
            count += 1
    print(f"Wrote {count} edges to {edges_path}")

    # seed.txt is written last so its presence marks the scrape as complete.
    with open(os.path.join(run_dir, "seed.txt"), "w", encoding="utf-8") as f:
        f.write(seed_id)

    # 2. Rank
    v, idx_to_id = math_engine.rank(edges_path, seed_id)
    order = math_engine.top_n(v, idx_to_id, config.TOP_N)

    with open(ranked_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "PaperID", "Score"])
        for rank, (pid, score) in enumerate(order, start=1):
            writer.writerow([rank, pid, f"{score:.8f}"])
    print(f"Wrote top {config.TOP_N} to {ranked_path}")

    # 3. Display
    rankings = load_rankings(ranked_path)
    paper_ids = [pid for _, pid, _ in rankings]
    by_id = client.get_papers(paper_ids)

    print("=" * 80)
    print(f" Personalized PageRank Reading List  (top {len(rankings)})")
    print(f" Run: {run_dir}")
    print("=" * 80)
    for rank, pid, score in rankings:
        print(format_entry(rank, score, by_id.get(pid)))
        print()


if __name__ == "__main__":
    main()
