"""Hull-gauge sweep over subset sizes k and subset-selection strategies (see sae_conditions.py).

For each of n_targets randomly chosen frequent features d, each strategy and each k, we pick S using
the real data's statistics and compute
    g(d, S)        the real hull gauge (condition holds iff <= 1)
    g_null(d, S)   the same under a null:
                     --null perm  each feature's activity shuffled independently across tokens
                                  (keeps marginals + dictionary, destroys co-activation structure)
                     --null cos   d's cosines swapped for those of a random other feature d2
Results go to CACHE_DIR/<dataset>/subset_sweep_<null>.npz.

Usage: python run_subset_sweep.py [--dataset openwebtext] [--null perm] [--n_targets 150] [--workers 10]
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from sae_conditions import FeatureData, choose_subset, hull_gauge, pairwise_check

CACHE_DIR = os.path.expanduser("~/sae_identifiability_cache")
KS = [1, 2, 3, 5, 10, 30, 100, 300]
STRATEGIES = ["random", "top_cos", "subordinate", "worst_pairs"]

fd = fd_perm = gauge = null = None  # set in main, inherited by forked workers


def run_target(args):
    d, seed = args
    rng = np.random.default_rng(seed)
    out = np.zeros((len(STRATEGIES), len(KS), 2))
    for si, strategy in enumerate(STRATEGIES):
        for ki, k in enumerate(KS):
            S = choose_subset(fd, d, k, strategy, rng, gauge)
            out[si, ki, 0] = hull_gauge(fd, d, S)
            if null == "perm":
                out[si, ki, 1] = hull_gauge(fd_perm, d, S)
            else:
                d2 = rng.choice(np.setdiff1d(np.arange(fd.F), np.append(S, d)))
                out[si, ki, 1] = hull_gauge(fd, d, S, cos_row=fd.cos[d2])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="openwebtext")
    parser.add_argument("--null", default="perm", choices=["perm", "cos"])
    parser.add_argument("--n_targets", type=int, default=150)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--freq_cutoff", type=float, default=0.005)
    args = parser.parse_args()
    null = args.null

    cache = os.path.join(CACHE_DIR, args.dataset)
    fd = FeatureData(sp.load_npz(f"{cache}/feature_acts.npz"), np.load(f"{cache}/sae_params.npz")["W_dec"], args.freq_cutoff)
    gauge, _, _ = pairwise_check(fd)

    rng = np.random.default_rng(0)
    fd_perm = fd.permuted(rng) if null == "perm" else None
    targets = rng.choice(fd.F, args.n_targets, replace=False)
    jobs = [(d, s) for d, s in zip(targets, rng.integers(1 << 31, size=len(targets)))]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(run_target, jobs), total=len(jobs)))

    out_path = f"{cache}/subset_sweep_{null}.npz"
    np.savez(
        out_path,
        results=np.stack(results),  # (n_targets, strategy, k, [real, null])
        targets=targets, ks=KS, strategies=STRATEGIES, feature_ids=fd.idx[targets],
    )
    print("saved", out_path)
