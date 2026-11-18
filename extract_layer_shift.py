"""Layer shift: apply the main (layer-8) SAE to the residual stream at other layers.

The analogue of dataset shift: the same SAE, evaluated on a representation it was not trained on.
Runs GPT2-small on exactly the tokens in CACHE_DIR/openwebtext/tokens.npy and, for each layer L,
saves CACHE_DIR/owt_layer<L>/ in the same format as a dataset (x.npy is the layer-L residual stream;
feature_acts.npz / sae_out.npy are the layer-8 SAE applied to it), so dataset_shift.py,
joint_violations.py and residual_condition.py run on it unchanged.

Usage: python extract_layer_shift.py [--layers 4 6 7 9 10]
"""
import argparse
import os
import shutil

import numpy as np
import scipy.sparse as sp
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

CACHE_DIR = os.path.expanduser("~/sae_identifiability_cache")
CTX = 128

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, nargs="+", default=[4, 6, 7, 9, 10])
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre", device=device)
model = HookedTransformer.from_pretrained("gpt2-small", device=device, **sae.cfg.metadata.model_from_pretrained_kwargs)
sae.eval()

src = f"{CACHE_DIR}/openwebtext"
tokens = np.load(f"{src}/tokens.npy").reshape(-1, CTX - 1)
tokens = torch.tensor(np.hstack([np.full((len(tokens), 1), model.tokenizer.bos_token_id), tokens]))

for L in args.layers:  # one layer at a time, to keep memory small
    hook = f"blocks.{L}.hook_resid_pre"
    acts, outs, xs = [], [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(tokens), 32), desc=f"layer {L}"):
            _, cache = model.run_with_cache(tokens[start : start + 32].to(device), names_filter=hook, stop_at_layer=L + 1)
            x = cache[hook][:, 1:]
            z = sae.encode(x)
            acts.append(sp.csr_matrix(z.reshape(-1, z.shape[-1]).cpu().numpy()))
            outs.append(sae.decode(z).reshape(-1, x.shape[-1]).cpu().numpy())
            xs.append(x.reshape(-1, x.shape[-1]).cpu().numpy())
    out = f"{CACHE_DIR}/owt_layer{L}"
    os.makedirs(out, exist_ok=True)
    feature_acts = sp.vstack(acts).tocsr()
    x, x_hat = np.concatenate(xs), np.concatenate(outs)
    sp.save_npz(f"{out}/feature_acts.npz", feature_acts)
    np.save(f"{out}/sae_out.npy", x_hat)
    np.save(f"{out}/x.npy", x)
    shutil.copy(f"{src}/tokens.npy", f"{out}/tokens.npy")
    fvu = ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
    print(f"layer {L}: mean L0 {feature_acts.getnnz(axis=1).mean():.1f}, FVU {fvu:.3f} -> {out}")
