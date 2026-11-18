"""Amortisation gap of the GPT2-small block-8 SAE (gpt2-small-res-jb) on the cached tokens of one dataset.
Usage: python amortisation_gap/amortisation_gap.py [dataset] [iters]   (default: openwebtext 800; ~30 min on an M-series GPU)

For the SAE's own decoder, solve the per-token nonnegative lasso (the SAE's training objective,
paper convention ||x - b - W^T z||^2 + 2 lam ||z||_1, lam = l1_coeff * d / 2 * ||x - mean x||) by FISTA,
warm-started at the encoder's codes. Saves the optimal codes in the same format as feature_acts.npz
(to <dataset>/optcodes/feature_acts.npz) so sae_conditions.py can be run on them unchanged."""
import json, os, sys, time
import numpy as np, scipy.sparse as sp, torch

DS = sys.argv[1] if len(sys.argv) > 1 else "openwebtext"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
CHUNK = 4096
C = os.path.expanduser(f"~/sae_identifiability_cache/{DS}")
P = os.path.expanduser("~/sae_identifiability_cache/openwebtext")  # SAE params (same SAE everywhere)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
L1 = 8e-5  # the SAE's training L1 coefficient

p = np.load(f"{P}/sae_params.npz")
xall = np.load(f"{C}/x.npy").astype(np.float32)
mu_train = np.load(f"{P}/x.npy").astype(np.float64).mean(0)
Zall = sp.load_npz(f"{C}/feature_acts.npz").tocsr()
W = torch.tensor(p["W_dec"], device=dev); b = torch.tensor(p["b_dec"], device=dev)
d = W.shape[1]
step = 1.0 / (2 * float(np.linalg.norm(p["W_dec"].astype(np.float64), 2)) ** 2)
lam_all = (L1 * d / 2 * np.linalg.norm(xall - mu_train, axis=1)).astype(np.float32)

rows, stats = [], {k: [] for k in ["mse0", "l10", "mse1", "l11", "nnz0", "nnz1", "both", "kkt"]}
t0 = time.time()
for s in range(0, xall.shape[0], CHUNK):
    x = torch.tensor(xall[s:s + CHUNK], device=dev); xc = x - b
    lam = torch.tensor(lam_all[s:s + CHUNK], device=dev)[:, None]
    Z0 = torch.tensor(Zall[s:s + CHUNK].toarray(), device=dev, dtype=torch.float32)
    Z, Y, t = Z0.clone(), Z0.clone(), 1.0
    for it in range(ITERS):
        Zn = torch.clamp(Y + step * 2 * (xc - Y @ W) @ W.T - step * 2 * lam, min=0)
        tn = (1 + (1 + 4 * t * t) ** 0.5) / 2
        Y = Zn + (t - 1) / tn * (Zn - Z); Z, t = Zn, tn
    Z = torch.where(Z > 1e-6, Z, torch.zeros_like(Z))
    for k, ZZ in (("0", Z0), ("1", Z)):
        stats["mse" + k].append(((xc - ZZ @ W) ** 2).sum(1).cpu().numpy())
        stats["l1" + k].append((2 * lam[:, 0] * ZZ.sum(1)).cpu().numpy())
        stats["nnz" + k].append((ZZ > 0).sum(1).cpu().numpy())
    stats["both"].append(((Z0 > 0) & (Z > 0)).sum(1).cpu().numpy())
    G = -2 * (xc - Z @ W) @ W.T
    stats["kkt"].append((torch.clamp(-(G + 2 * lam), min=0) / (2 * lam)).max(1).values.cpu().numpy())
    rows.append(sp.csr_matrix(Z.cpu().numpy()))
    print(f"{s + CHUNK}/{xall.shape[0]}  {time.time() - t0:.0f}s", flush=True)

st = {k: np.concatenate(v) for k, v in stats.items()}
os.makedirs(f"{C}/optcodes", exist_ok=True)
sp.save_npz(f"{C}/optcodes/feature_acts.npz", sp.vstack(rows).tocsr())
np.savez(f"{C}/optcodes/gap_stats.npz", **st)
var = float(((xall - xall.mean(0)) ** 2).sum(1).mean())
tot0, tot1 = st["mse0"] + st["l10"], st["mse1"] + st["l11"]
summary = dict(
    dataset=DS, n_tokens=int(len(tot0)), iters=ITERS,
    obj_enc=float(tot0.mean()), obj_opt=float(tot1.mean()),
    rel_gap_mean=float(((tot0 - tot1) / tot1).mean()), rel_gap_median=float(np.median((tot0 - tot1) / tot1)),
    fvu_enc=float(st["mse0"].mean() / var), fvu_opt=float(st["mse1"].mean() / var),
    l1_enc=float(st["l10"].mean()), l1_opt=float(st["l11"].mean()),
    l0_enc=float(st["nnz0"].mean()), l0_opt=float(st["nnz1"].mean()),
    frac_enc_support_kept=float(st["both"].sum() / st["nnz0"].sum()),
    frac_opt_support_in_enc=float(st["both"].sum() / st["nnz1"].sum()),
    kkt_max_rel=float(st["kkt"].max()), kkt_p99_rel=float(np.quantile(st["kkt"], 0.99)),
)
json.dump(summary, open(f"{C}/optcodes/gap_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
