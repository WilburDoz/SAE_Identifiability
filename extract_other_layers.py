"""Encode the cached OpenWebText sample with the gpt2-small-res-jb SAEs at other layers.

Uses exactly the tokens in CACHE_DIR/openwebtext/tokens.npy (BOS is re-added before running the model
and dropped again afterwards), so latents from different layers line up token by token.
Saves CACHE_DIR/openwebtext/layer_<L>/{feature_acts.npz, sae_params.npz}.

Usage: python extract_other_layers.py [--layers 4 6 10]
"""
import argparse
import os

import numpy as np
import scipy.sparse as sp
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

CACHE_DIR = os.path.expanduser("~/sae_identifiability_cache")
CTX = 128  # the SAEs' context size; tokens.npy holds CTX - 1 tokens per context (BOS dropped)

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, nargs="+", default=[4, 6, 10])
parser.add_argument("--dataset", default="openwebtext")
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
saes = {L: SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=f"blocks.{L}.hook_resid_pre", device=device)
        for L in args.layers}
model = HookedTransformer.from_pretrained("gpt2-small", device=device,
                                          **saes[args.layers[0]].cfg.metadata.model_from_pretrained_kwargs)
hooks = [f"blocks.{L}.hook_resid_pre" for L in args.layers]

tokens = np.load(f"{CACHE_DIR}/{args.dataset}/tokens.npy").reshape(-1, CTX - 1)
bos = np.full((len(tokens), 1), model.tokenizer.bos_token_id)
tokens = torch.tensor(np.hstack([bos, tokens]))

blocks = {L: [] for L in args.layers}
with torch.no_grad():
    for start in tqdm(range(0, len(tokens), 32)):
        _, cache = model.run_with_cache(tokens[start : start + 32].to(device), names_filter=hooks,
                                        stop_at_layer=max(args.layers) + 1)
        for L, hook in zip(args.layers, hooks):
            z = saes[L].encode(cache[hook][:, 1:])
            blocks[L].append(sp.csr_matrix(z.reshape(-1, z.shape[-1]).cpu().numpy()))

for L in args.layers:
    out = f"{CACHE_DIR}/{args.dataset}/layer_{L}"
    os.makedirs(out, exist_ok=True)
    acts = sp.vstack(blocks[L]).tocsr()
    sp.save_npz(f"{out}/feature_acts.npz", acts)
    np.savez(f"{out}/sae_params.npz", W_dec=saes[L].W_dec.detach().cpu().numpy())
    print(f"layer {L}: {acts.shape[0]} tokens, mean L0 {acts.getnnz(axis=1).mean():.1f} -> {out}")
