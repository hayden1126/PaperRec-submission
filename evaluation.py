"""Evaluation methods for Personalized PageRank: convergence and sensitivity analysis."""

import numpy as np
from scipy.sparse import csr_matrix

import math_engine


def power_iteration(P: csr_matrix,
    seed_idx: int,
    damping: float,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    """Run v_{k+1} = dPv_k + (1-d)e_s; return final v and per-iteration L1 residuals."""
    n = P.shape[0]
    e_s = np.zeros(n, dtype=np.float64)
    e_s[seed_idx] = 1.0
    v = e_s.copy()
    residuals: list[float] = []
    for _ in range(max_iter):
        v_next = damping * (P @ v) + (1.0 - damping) * e_s
        residuals.append(float(np.linalg.norm(v_next - v, ord=1)))
        v = v_next
        if residuals[-1] < tol:
            break
    return v, residuals


def damping_sweep(P: csr_matrix,
    seed_idx: int,
    damping_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PPR at each damping and pairwise Pearson correlation. Returns (corr, scores)."""
    k = len(damping_values)
    n = P.shape[0]
    scores = np.zeros((k, n), dtype=np.float64)
    for i, d in enumerate(damping_values):
        scores[i] = math_engine.personalized_pagerank(P, seed_idx, float(d))

    corr = np.corrcoef(scores)
    return corr, scores
