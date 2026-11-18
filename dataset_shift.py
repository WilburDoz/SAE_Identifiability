"""Pairwise feature-feature condition (eq. 6) on the same SAE across datasets.

The SAE was trained on OpenWebText. The theory says the condition must hold on the training
distribution; off-distribution it can fail, and the prediction is *how*: through hierarchies or
never-co-inactive pairs that appear when some features become much denser in the new domain.

For each dataset in CACHE_DIR we save, to CACHE_DIR/dataset_shift_<sae>.json:
    summary    FVU, mean L0, number of frequent features, violation counts by type, gauge quantiles
    violations one row per violating ordered pair (d, d'): feature ids, frequencies here and on OWT,
               cosine, type, child's top tokens, and whether it replicates in both halves of the data

Usage: python dataset_shift.py [--sae fs_768] [--datasets openwebtext pile code ...]
"""
import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
from transformers import AutoTokenizer

from sae_conditions import FeatureData, pairwise_check

CACHE_DIR = os.path.expanduser("~/sae_identifiability_cache")
ALL_DATASETS = ["openwebtext", "openwebtext_b", "c4", "wiki", "pile", "code", "code_python", "wiki_de", "wiki_fr"]


def pair_violates(Z_half, W_hat, a, b):
    """Brute-force pairwise test for ordered pair (a, b) (SAE feature ids) on a subset of tokens."""
    za, zb = Z_half[:, a].toarray().ravel(), Z_half[:, b].toarray().ravel()
    if za.sum() == 0 or zb.sum() == 0:
        return None  # feature absent from this half: can't say
    ub = zb / zb.mean()
    t = 1 - W_hat[a] @ W_hat[b]
    off = za == 0
    return bool(t > ub[off].max() * (1 + 1e-3) or t < ub[off].min() - 1e-3)


def analyse(dataset, sae, W_dec, freq_owt, tokenizer, freq_cutoff):
    c = f"{CACHE_DIR}/{dataset}"
    Z = sp.load_npz(f"{c}/{sae}/feature_acts.npz").tocsc()
    x, x_hat = np.load(f"{c}/x.npy"), np.load(f"{c}/{sae}/sae_out.npy")
    tokens = np.load(f"{c}/tokens.npy")
    fvu = float(((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
    del x, x_hat

    fd = FeatureData(Z, W_dec, freq_cutoff)
    gauge, violations, M = pairwise_check(fd)
    freq = fd.n_active / fd.N
    finite = np.isfinite(gauge)

    both_off = fd.N - fd.n_active[:, None] - fd.n_active[None, :] + fd.coactive
    np.fill_diagonal(both_off, 1)
    half = fd.N // 2
    W_hat = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)

    rows = []
    for d, dp in zip(*np.where(violations)):
        a, b = int(fd.idx[d]), int(fd.idx[dp])
        kind = "never_coinactive" if both_off[d, dp] == 0 else ("strict_hierarchy" if M[d, dp] == 0 else "upper_end")
        on_b = Z[:, b].nonzero()[0]
        top = on_b[np.argsort(-Z[on_b, b].toarray().ravel())[:10]]
        rows.append(dict(
            d=a, d_prime=b, kind=kind, cos=float(fd.cos[d, dp]),
            freq_d=float(freq[d]), freq_d_prime=float(freq[dp]),
            freq_d_owt=float(freq_owt[a]), freq_d_prime_owt=float(freq_owt[b]),
            n_on_d_prime=int(fd.n_active[dp]),
            d_prime_top_tokens=[tokenizer.decode([t]) for t in tokens[top]],
            replicates=[pair_violates(Z[:half], W_hat, a, b), pair_violates(Z[half:], W_hat, a, b)],
        ))

    kinds = [r["kind"] for r in rows]
    summary = dict(
        dataset=dataset, sae=sae or "main_24576", fvu=fvu, mean_l0=float(Z.getnnz() / fd.N), n_features=int(fd.F),
        n_dense_10pct=int((freq > 0.1).sum()),
        n_violations=len(rows), **{f"n_{k}": kinds.count(k) for k in ["strict_hierarchy", "upper_end", "never_coinactive"]},
        n_violating_features=len({r["d"] for r in rows} | {r["d_prime"] for r in rows}),
        n_replicating=sum(all(r["replicates"]) for r in rows),
        n_pairs_gauge_above_0p1=int((gauge[finite] > 0.1).sum()),
        gauge_quantiles={q: float(np.percentile(gauge[finite], q)) for q in [50, 99, 99.99]},
        gauge_max_finite=float(gauge[finite].max()),
    )
    return summary, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--freq_cutoff", type=float, default=0.005)
    parser.add_argument("--sae", default="", help='"" for the main SAE, or fs_768, fs_1536, ...')
    args = parser.parse_args()

    W_dec = np.load(f"{CACHE_DIR}/openwebtext/{args.sae}/sae_params.npz")["W_dec"]
    Z_owt = sp.load_npz(f"{CACHE_DIR}/openwebtext/{args.sae}/feature_acts.npz")
    freq_owt = Z_owt.getnnz(axis=0) / Z_owt.shape[0]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    out_path = f"{CACHE_DIR}/dataset_shift_{args.sae or 'main'}.json"
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for ds in args.datasets:
        summary, rows = analyse(ds, args.sae, W_dec, freq_owt, tokenizer, args.freq_cutoff)
        results[ds] = dict(summary=summary, violations=rows)
        print(json.dumps(summary), flush=True)
        json.dump(results, open(out_path, "w"), indent=1)
