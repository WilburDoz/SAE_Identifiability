"""Search for violations of the feature-feature condition (eq. 6) that no single pair reveals.

For the n_targets features whose pairwise constraints are closest to binding, we test eq. 6 jointly
against the k features that are individually closest to violating (strategy "worst_pairs"), for
several k. Pairs that already violate on their own are excluded, so every violation found here is one
that no pairwise test can see. For every violating (d, S) we greedily shrink S to a minimal violating set, and re-test that
set on each half of the tokens to check it replicates.

Saves CACHE_DIR/joint_violations_<sae>.json with, per dataset:
    gauges     {k: [gauge for each target]}
    violations one entry per violating target: feature ids of d and the minimal S, gauges, replication

Usage: python joint_violations.py [--sae ""] [--datasets openwebtext code ...]
"""
import argparse
import json
import os

import numpy as np
import scipy.sparse as sp

from sae_conditions import FeatureData, choose_subset, hull_gauge, pairwise_check

CACHE_DIR = os.path.expanduser("~/sae_identifiability_cache")
KS = [1, 3, 10, 30, 100]


def minimal_violating_set(fd, d, S):
    """Greedily drop features from S while the joint condition stays violated."""
    S = list(S)
    changed = True
    while changed:
        changed = False
        for s in list(S):
            T = [x for x in S if x != s]
            if T and hull_gauge(fd, d, T) > 1:
                S, changed = T, True
    return S


def gauge_on_tokens(Z, W_dec, rows, d_id, S_ids):
    """Gauge for (d, S), given as SAE feature ids, recomputed using only the tokens in `rows`."""
    fd = FeatureData(Z[rows][:, [d_id] + S_ids], W_dec[[d_id] + S_ids], freq_cutoff=0.0)
    if fd.F < len(S_ids) + 1:
        return None  # some feature never fires in this half
    pos = {f: i for i, f in enumerate([d_id] + S_ids)}
    order = {orig: new for new, orig in enumerate(fd.idx)}  # FeatureData reorders by frequency
    return hull_gauge(fd, order[pos[d_id]], [order[pos[s]] for s in S_ids])


def search(dataset, sae, n_targets, freq_cutoff):
    c = f"{CACHE_DIR}/{dataset}/{sae}"
    Z = sp.load_npz(f"{c}/feature_acts.npz").tocsr()
    W_dec = np.load(f"{CACHE_DIR}/openwebtext/{sae}/sae_params.npz")["W_dec"]
    fd = FeatureData(Z, W_dec, freq_cutoff)
    pair_gauge, _, _ = pairwise_check(fd)
    # Only build sets from pairs that pass on their own, so every violation found is genuinely joint
    pair_gauge[pair_gauge > 1] = np.nan
    closest = np.nanmax(np.where(np.isfinite(pair_gauge), pair_gauge, np.nan), axis=1)
    targets = np.argsort(-closest)[:n_targets]
    rng = np.random.default_rng(0)  # unused by worst_pairs, required by choose_subset's signature

    gauges = {k: [] for k in KS}
    violations = []
    halves = [np.arange(fd.N // 2), np.arange(fd.N // 2, fd.N)]
    for d in targets:
        found = False
        for k in KS:
            S = choose_subset(fd, d, k, "worst_pairs", rng, pair_gauge)
            g = hull_gauge(fd, d, S)
            gauges[k].append(g)
            if g > 1 and not found:  # record the smallest k at which d violates
                found = True
                S_min = minimal_violating_set(fd, d, S)
                d_id, S_ids = int(fd.idx[d]), [int(fd.idx[s]) for s in S_min]
                violations.append(dict(
                    d=d_id, k=k, gauge=g, minimal_S=S_ids, minimal_gauge=hull_gauge(fd, d, S_min),
                    max_pair_gauge_in_S=float(np.max(pair_gauge[d, S_min])),
                    half_gauges=[gauge_on_tokens(Z, W_dec, h, d_id, S_ids) for h in halves],
                ))
    return dict(targets=[int(fd.idx[d]) for d in targets], gauges={str(k): v for k, v in gauges.items()},
                violations=violations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["openwebtext", "openwebtext_b", "pile", "wiki", "code", "code_python"])
    parser.add_argument("--sae", default="")
    parser.add_argument("--n_targets", type=int, default=25)
    parser.add_argument("--freq_cutoff", type=float, default=0.005)
    args = parser.parse_args()

    out_path = f"{CACHE_DIR}/joint_violations_{args.sae or 'main'}.json"
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for ds in args.datasets:
        results[ds] = search(ds, args.sae, args.n_targets, args.freq_cutoff)
        n = len(results[ds]["violations"])
        print(f"{ds}: {n} of {args.n_targets} targets violate jointly; median gauge by k:",
              {k: round(float(np.median(v)), 3) for k, v in results[ds]["gauges"].items()}, flush=True)
        json.dump(results, open(out_path, "w"), indent=1)
