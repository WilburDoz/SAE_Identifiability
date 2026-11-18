"""Run the eq. (6) pairwise check and the eq. (8) residual check on the optimal codes from amortisation_gap.py
(same SAE decoder), alongside the encoder codes, on one dataset.
Usage: python amortisation_gap/conditions_optimal_codes.py [dataset]   (run amortisation_gap.py first)"""
import json, os, sys
import numpy as np, scipy.sparse as sp
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers import AutoTokenizer
import dataset_shift, residual_condition

DS = sys.argv[1] if len(sys.argv) > 1 else "openwebtext"
C = os.path.expanduser(f"~/sae_identifiability_cache/{DS}")
P = os.path.expanduser("~/sae_identifiability_cache/openwebtext")
p = np.load(f"{P}/sae_params.npz")
if not os.path.exists(f"{C}/optcodes/sae_params.npz"):  # the analysis modules look for the SAE here
    os.symlink(f"{P}/sae_params.npz", f"{C}/optcodes/sae_params.npz")
if not os.path.exists(f"{C}/optcodes/sae_out.npy"):
    Z = sp.load_npz(f"{C}/optcodes/feature_acts.npz")
    np.save(f"{C}/optcodes/sae_out.npy", (Z @ p["W_dec"] + p["b_dec"]).astype(np.float32))
W_dec = p["W_dec"]
Z_owt = sp.load_npz(f"{P}/feature_acts.npz"); freq_owt = Z_owt.getnnz(axis=0) / Z_owt.shape[0]
tok = AutoTokenizer.from_pretrained("gpt2")
lam = residual_condition.training_lambda()
out = {}
for sae in ["", "optcodes"]:
    s, rows = dataset_shift.analyse(DS, sae, W_dec, freq_owt, tok, 0.005)
    r = residual_condition.analyse(DS, sae, lam, tok)
    r_small = {k: v for k, v in r.items() if k not in ("violations", "pc_sd", "pc_top_tokens")}
    out[sae or "encoder"] = dict(eq6=s, eq6_violations=rows[:20], eq8=r_small)
    print(sae or "encoder", json.dumps(s), json.dumps(r_small), sep="\n", flush=True)
json.dump(out, open(f"{C}/optcodes/conditions.json", "w"), indent=1)
