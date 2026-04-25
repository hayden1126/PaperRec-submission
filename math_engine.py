"""Personalized PageRank on a scraped citation edge list."""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve

import config


def load_edges(path: str) -> pd.DataFrame:
    """Read edge list CSV and clean out invalid rows."""
    df = pd.read_csv(path, dtype=str)
    df = df.dropna() # remove broken data
    df = df[df["Source"] != df["Target"]] # remove self loops
    return df


def build_index(df: pd.DataFrame) -> dict[str, int]:
    """Map every unique paper ID to a contiguous integer index."""
    unique_ids = pd.unique(pd.concat([df["Source"], df["Target"]], ignore_index=True))  # collect all IDs from both columns
    return {pid: i for i, pid in enumerate(unique_ids)}


def build_transition_matrix_for_seed(df: pd.DataFrame,
    id_to_idx: dict[str, int],
    seed_idx: int,
) -> csr_matrix:
    """Column-stochastic transition matrix. Dangling columns are redirected to the
    given seed row, so the returned matrix is only valid for PPR with this seed."""
    n = len(id_to_idx)
    rows = df["Target"].map(id_to_idx).to_numpy()
    cols = df["Source"].map(id_to_idx).to_numpy()
    data = np.ones(len(df), dtype=np.float64)

    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()  # sparse adjacency matrix from edge list

    col_sums = np.asarray(A.sum(axis=0)).ravel().astype(np.float64)  # out-degree of each node
    inv = np.zeros(n, dtype=np.float64)
    nz = col_sums > 0
    inv[nz] = 1.0 / col_sums[nz]  # inverse degrees, zero for dangling nodes
    P = A @ diags(inv)  # normalize columns so each non-dangling column sums to 1

    dangling = np.where(~nz)[0]  # indices of columns with no outgoing edges
    if dangling.size > 0:
        seed_rows = np.full(dangling.size, seed_idx)
        fix = coo_matrix(
            (np.ones(dangling.size), (seed_rows, dangling)),
            shape=(n, n),
        )  # rank-one update placing all mass of each dangling column on the seed row
        P = P + fix  # teleport dangling nodes back to the seed, restoring column-stochasticity

    return P.tocsr()


def personalized_pagerank(P: csr_matrix, seed_idx: int, damping: float) -> np.ndarray:
    """Solve the PPR linear system and return v, the stationary distribution."""
    n = P.shape[0]
    e_s = np.zeros(n, dtype=np.float64)
    e_s[seed_idx] = 1.0  # teleport vector: all weight on the seed node

    A_solve = (eye(n, format="csr") - damping * P).tocsc()  # (I - αP), rearranged from v = αPv + (1-α)e_s
    b = (1.0 - damping) * e_s
    # No post-normalization: P is column-stochastic, so v sums to 1 by construction.
    return spsolve(A_solve, b)


def rank(edges_path: str, seed_id: str) -> tuple[np.ndarray, dict[int, str]]:
    """Load edges, build the graph, run PPR from seed, and return scores with ID mapping."""
    df = load_edges(edges_path)
    id_to_idx = build_index(df)
    print(f"Loaded {len(df)} edges over {len(id_to_idx)} unique nodes")

    if seed_id not in id_to_idx:
        raise SystemExit(f"Seed {seed_id} not present in {edges_path}")

    seed_idx = id_to_idx[seed_id]
    P = build_transition_matrix_for_seed(df, id_to_idx, seed_idx)
    v = personalized_pagerank(P, seed_idx, config.DAMPING)
    idx_to_id = {i: pid for pid, i in id_to_idx.items()}
    return v, idx_to_id


def top_n(v: np.ndarray, idx_to_id: dict[int, str], n: int) -> list[tuple[str, float]]:
    """Return the top-n (paper_id, score) pairs sorted by score descending."""
    order = np.argsort(-v)[:n]  # indices of the n highest scores
    return [(idx_to_id[int(i)], float(v[int(i)])) for i in order]
