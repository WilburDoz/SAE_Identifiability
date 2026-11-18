"""Feature-residual condition (eqs. 7-8) on real SAEs.

Take a principal direction u of the (demeaned) residuals E = X - X_hat, with standard deviation sigma and
per-token score s = E u. Eq. (7) with the perturbation delta_z = s gives, exactly and for any residual:

    -sigma^2 cos(w_d, u)  in  lambda * [min_{A_d} s, max_{A_d} s]            (eq. 8, full-rank version)

i.e. a feature may point towards a high-variance direction left in the residual only if the residual
takes extreme values there while the feature is off. Rather than fixing lambda, for every (d, u) we
compute the smallest lambda at which the pair passes,

    lambda_min(d, u) = sigma^2 |cos| / (max_{A_d} s  if cos < 0  else  |min_{A_d} s|),

and lambda* = max over pairs: the dictionary satisfies eq. 8 for every lambda >= lambda*. We compare
lambda* against the SAE's training lambda. Converting SAE-lens's loss (per-element MSE normalised by the
centred input norm, plus l1_coefficient * sum|z|) to the paper's convention,
||x - x_hat||^2 + 2 lambda ||z||_1, gives lambda = l1_coefficient * d_in / 2 * ||x - mean(x)||; we use the
median over OpenWebText tokens.

Saves CACHE_DIR/residual_condition.json. Usage: python residual_condition.py [--saes "" fs_768 ...]
"""
import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
from transformers import AutoTokenizer

from sae_conditions import FeatureData

CACHE_DIR = os.path.expanduser("~/sae_identifiability_cache")
L1_COEFFICIENT = 8e-5  # from the SAEs' training cfg.json (same for every width)
N_PCS = 50


def training_lambda():
    x = np.load(f"{CACHE_DIR}/openwebtext/x.npy").astype(np.float64)
    return float(np.median(L1_COEFFICIENT * x.shape[1] / 2 * np.linalg.norm(x - x.mean(0), axis=1)))


def analyse(dataset, sae, lam_train, tokenizer, freq_cutoff=0.005):
    c = f"{CACHE_DIR}/{dataset}"
    x = np.load(f"{c}/x.npy").astype(np.float64)
    E = x - np.load(f"{c}/{sae}/sae_out.npy")
    E -= E.mean(0)
    fd = FeatureData(sp.load_npz(f"{c}/{sae}/feature_acts.npz"), np.load(f"{CACHE_DIR}/openwebtext/{sae}/sae_params.npz")["W_dec"], freq_cutoff)

    var, U = np.linalg.eigh(E.T @ E / len(E))
    var, U = var[::-1][:N_PCS], U[:, ::-1][:, :N_PCS]
    S = E @ U                       # (N, N_PCS) residual scores
    cos = fd.W_hat @ U              # (F, N_PCS)

    lam_min = np.zeros_like(cos)
    for d in range(fd.F):
        off = np.ones(fd.N, bool)
        off[fd.U.indices[fd.U.indptr[d] : fd.U.indptr[d + 1]]] = False
        s_max, s_min = S[off].max(0), S[off].min(0)
        lam_min[d] = var * np.abs(cos[d]) / np.where(cos[d] < 0, s_max, -s_min)

    tokens = np.load(f"{c}/tokens.npy")
    def top_tokens(j, sign):  # tokens at which residual direction j is most extreme, with 3 tokens of context
        idx = np.argsort(-sign * S[:, j])[:8]
        return [tokenizer.decode(tokens[max(i - 3, 0) : i + 1]) for i in idx]

    viol = np.argwhere(lam_min > lam_train)
    return dict(
        dataset=dataset, sae=sae or "main_24576", lam_train=lam_train, lam_star=float(lam_min.max()),
        pc_sd=np.sqrt(var).tolist(),
        n_violations=int(len(viol)), n_pairs=int(lam_min.size),
        violations=[dict(feature=int(fd.idx[d]), pc=int(j), cos=float(cos[d, j]), lam_min=float(lam_min[d, j]),
                         feature_freq=float(fd.n_active[d] / fd.N)) for d, j in viol],
        pc_top_tokens={int(j): dict(positive=top_tokens(j, 1), negative=top_tokens(j, -1))
                       for j in sorted({int(j) for _, j in viol} | {0})},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saes", nargs="+", default=["", "fs_768", "fs_1536", "fs_3072", "fs_6144"])
    parser.add_argument("--datasets", nargs="+", default=["openwebtext", "openwebtext_b", "c4", "wiki", "pile", "code", "code_python"])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    lam_train = training_lambda()
    out_path = f"{CACHE_DIR}/residual_condition.json"
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for sae in args.saes:
        for ds in args.datasets:
            r = analyse(ds, sae, lam_train, tokenizer)
            results[f"{r['sae']}/{ds}"] = r
            print(f"{r['sae']:11s} {ds:14s} PC1 sd {r['pc_sd'][0]:6.2f}  lambda* {r['lam_star']:6.2f} "
                  f"(training lambda {lam_train:.2f})  violations {r['n_violations']}/{r['n_pairs']}", flush=True)
            json.dump(results, open(out_path, "w"), indent=1)
