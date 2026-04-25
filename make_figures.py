"""Generate paper figures (convergence plot, damping heatmap) from a pipeline run."""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
import evaluation
import math_engine


def load_run(run_dir: str) -> tuple:
    """Load the edge list and resolved seed id from a run directory."""
    df = math_engine.load_edges(os.path.join(run_dir, config.EDGES_CSV))
    with open(os.path.join(run_dir, "seed.txt"), encoding="utf-8") as f:
        seed_id = f.read().strip()
    return df, seed_id


def plot_convergence(residuals: list,
    damping: float,
    out_path: str,
) -> None:
    """Semi-log plot of ||v_k - v_{k-1}||_1 vs k, with theoretical geometric decay."""
    k_vals = np.arange(1, len(residuals) + 1)
    theory = residuals[0] * (damping ** (k_vals - 1))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(k_vals, residuals, marker="o", markersize=3, linewidth=1, label="empirical")
    ax.semilogy(k_vals, theory, linestyle="--", linewidth=1,
                label=rf"theory: $r_k = r_1 \cdot d^{{k-1}}$, $d={damping}$")
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"$\|v_k - v_{k-1}\|_1$")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmap(corr: np.ndarray,
    damping_values: np.ndarray,
    out_path: str,
) -> None:
    """Heatmap of Pearson correlation between PPR vectors across damping values."""
    labels = [f"{d:.2f}" for d in damping_values]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": r"Pearson $\rho$"},
        ax=ax,
    )
    ax.set_xlabel(r"damping $d$")
    ax.set_ylabel(r"damping $d$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def top_k_overlap(v: np.ndarray, in_degree: np.ndarray, k: int) -> int:
    """Count papers appearing in both top-k lists (PPR vs. citation count)."""
    return len(set(np.argsort(-v)[:k]) & set(np.argsort(-in_degree)[:k]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate paper figures from a pipeline run.")
    ap.add_argument("--run-dir", required=True, help="Path to a run directory from main.py")
    args = ap.parse_args()

    df, seed_id = load_run(args.run_dir)
    id_to_idx = math_engine.build_index(df)
    if seed_id not in id_to_idx:
        raise SystemExit(f"Seed {seed_id} not present in {args.run_dir}")
    seed_idx = id_to_idx[seed_id]
    P = math_engine.build_transition_matrix_for_seed(df, id_to_idx, seed_idx)
    n = P.shape[0]
    print(f"Loaded {len(df)} edges over {n} unique nodes (seed: {seed_id})")

    # Figure 1: convergence of power iteration at the default damping.
    _, residuals = evaluation.power_iteration(P, seed_idx, config.DAMPING)
    fig1_path = os.path.join(args.run_dir, "fig1_convergence.pdf")
    plot_convergence(residuals, config.DAMPING, fig1_path)
    ratios = [residuals[k + 1] / residuals[k] for k in range(len(residuals) - 1)]
    mean_ratio = float(np.mean(ratios[5:])) if len(ratios) > 5 else float(np.mean(ratios))
    print(f"Figure 1 saved: {fig1_path}")
    print(f"  converged in {len(residuals)} iterations; mean r_{{k+1}}/r_k = {mean_ratio:.4f} (theory: {config.DAMPING})")

    # Figure 2: damping sensitivity via Pearson correlation between score vectors.
    damping_values = np.array([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99])
    corr, scores = evaluation.damping_sweep(P, seed_idx, damping_values)
    fig2_path = os.path.join(args.run_dir, "fig2_damping.pdf")
    plot_heatmap(corr, damping_values, fig2_path)
    print(f"Figure 2 saved: {fig2_path}")

    # Baseline: top-N overlap between PPR at the default damping and raw citation count.
    in_degree = np.zeros(n, dtype=np.int64)
    for target in df["Target"]:
        in_degree[id_to_idx[target]] += 1
    v_default = scores[int(np.where(np.isclose(damping_values, config.DAMPING))[0][0])]
    overlap = top_k_overlap(v_default, in_degree, config.TOP_N)
    print(f"Top-{config.TOP_N} overlap (PPR vs. citation count): {overlap} of {config.TOP_N}")


if __name__ == "__main__":
    main()
