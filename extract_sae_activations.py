"""Run GPT2-small + the block-8 SAE-lens SAE over OpenWebText and cache everything
the downstream checks need, so analysis notebooks never have to re-run the model.

Saved to CACHE_DIR/<dataset_tag>/:
    feature_acts.npz   scipy CSR matrix, (num_tokens, d_sae), SAE latents z (BOS removed)
    x.npy              (num_tokens, d_in) SAE input (residual stream)
    sae_out.npy        (num_tokens, d_in) SAE reconstruction
    tokens.npy         (num_tokens,) token ids, for looking features up afterwards
    sae_params.npz     W_dec, b_dec, W_enc, b_enc
    fs_<width>/        feature_acts.npz, sae_out.npy, sae_params.npz for each feature-splitting SAE width
                       (same layer and training recipe, 768 ... 6144 latents); x.npy/tokens.npy are shared

Usage: python extract_sae_activations.py [--n_seqs 512] [--dataset openwebtext|pile|code|...] [--keep_bos]
"""
import argparse
import os

import numpy as np
import scipy.sparse as sp
import torch
from datasets import load_dataset
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

CACHE_DIR = os.path.expanduser("~/sae_identifiability_cache")
# name: (HF path, config, shuffle seed, document filter). openwebtext_b is a second, disjoint OWT
# sample: the sampling-noise baseline for any comparison against openwebtext.
DATASETS = {
    "openwebtext": ("Skylion007/openwebtext", None, 0, None),
    "openwebtext_b": ("Skylion007/openwebtext", None, 1, None),
    "c4": ("NeelNanda/c4-10k", None, 0, None),
    "wiki": ("NeelNanda/wiki-10k", None, 0, None),
    "pile": ("NeelNanda/pile-10k", None, 0, None),
    "code": ("NeelNanda/code-10k", None, 0, None),
    "code_python": ("NeelNanda/code-10k", None, 0, lambda doc: doc["path"].endswith(".py")),  # a narrow domain
    # Streaming wikimedia/wikipedia through `datasets` stalls, so run_real_sae.sh downloads one parquet
    # shard per language (train-00002, ~350MB) to CACHE_DIR/raw/ and we read it locally
    "wiki_de": ("parquet", f"{CACHE_DIR}/raw/wiki_de.parquet", 0, None),
    "wiki_fr": ("parquet", f"{CACHE_DIR}/raw/wiki_fr.parquet", 0, None),
    # held-out Python (codeparrot-clean shard 1) for the code-trained SAE; shard 0 is its training data
    "codeparrot_eval": ("parquet", f"{CACHE_DIR}/raw/codeparrot_1.parquet", 0, None),
}
MAX_DOC_CHARS = 4000  # truncate long documents so each sample spans many of them

parser = argparse.ArgumentParser()
parser.add_argument("--n_seqs", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dataset", default="openwebtext", choices=DATASETS)
parser.add_argument("--shuffle_buffer", type=int, default=5000)
parser.add_argument("--widths", type=int, nargs="*", default=[768, 1536, 3072, 6144],
                    help="also encode with these gpt2-small-res-jb-feature-splitting SAEs (saved under fs_<width>/)")
parser.add_argument("--keep_bos", action="store_true", help="keep BOS positions (as the original SAE_Real notebook did)")
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
out_dir = os.path.join(CACHE_DIR, args.dataset + ("_bos" if args.keep_bos else ""))
os.makedirs(out_dir, exist_ok=True)

# The SAE was trained on centred-writing-weights GPT2, so load the model the same way.
# saes maps output subdirectory -> SAE; "" is the main 24576-latent SAE used throughout
saes = {"": SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.8.hook_resid_pre", device=device)}
for w in args.widths:
    saes[f"fs_{w}"] = SAE.from_pretrained(
        release="gpt2-small-res-jb-feature-splitting", sae_id=f"blocks.8.hook_resid_pre_{w}", device=device
    )
sae = saes[""]
model = HookedTransformer.from_pretrained("gpt2-small", device=device, **sae.cfg.metadata.model_from_pretrained_kwargs)
for s in saes.values():
    s.eval()
hook_name = sae.cfg.metadata.hook_name
ctx = sae.cfg.metadata.context_size

# Stream just enough documents to fill n_seqs contexts (openwebtext is ~40GB, don't download it all)
path, config, seed, doc_filter = DATASETS[args.dataset]
if path == "parquet":  # a local file
    stream = load_dataset("parquet", data_files=config, split="train", streaming=True)
else:
    stream = load_dataset(path, config, split="train", streaming=True)
if doc_filter is not None:
    stream = stream.filter(doc_filter)
stream = stream.shuffle(seed=seed, buffer_size=args.shuffle_buffer)
docs = []
n_chars = 0
for doc in stream:
    docs.append((doc["text"] if "text" in doc else doc["content"])[:MAX_DOC_CHARS])  # codeparrot calls it "content"
    n_chars += len(docs[-1])
    if n_chars > args.n_seqs * ctx * 6:  # ~4 chars/token, with margin
        break
from datasets import Dataset
token_dataset = tokenize_and_concatenate(
    dataset=Dataset.from_dict({"text": docs}),
    tokenizer=model.tokenizer,
    max_length=ctx,
    add_bos_token=sae.cfg.metadata.prepend_bos,
)
all_tokens = token_dataset["tokens"][: args.n_seqs]
assert all_tokens.shape[0] == args.n_seqs, f"only got {all_tokens.shape[0]} sequences"

acts_blocks = {k: [] for k in saes}
outs = {k: [] for k in saes}
xs, toks = [], []
with torch.no_grad():
    for start in tqdm(range(0, args.n_seqs, args.batch_size)):
        batch = all_tokens[start : start + args.batch_size].to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_name, stop_at_layer=9)
        first = 0 if args.keep_bos else 1  # BOS is a huge-norm outlier the SAE wasn't really fit to
        x = cache[hook_name][:, first:]
        xs.append(x.reshape(-1, x.shape[-1]).cpu().numpy())
        toks.append(batch[:, first:].reshape(-1).cpu().numpy())
        for k, s in saes.items():
            z = s.encode(x)
            acts_blocks[k].append(sp.csr_matrix(z.reshape(-1, z.shape[-1]).cpu().numpy()))
            outs[k].append(s.decode(z).reshape(-1, x.shape[-1]).cpu().numpy())

np.save(os.path.join(out_dir, "x.npy"), np.concatenate(xs))
np.save(os.path.join(out_dir, "tokens.npy"), np.concatenate(toks))
for k, s in saes.items():
    d = os.path.join(out_dir, k)
    os.makedirs(d, exist_ok=True)
    feature_acts = sp.vstack(acts_blocks[k]).tocsr()
    sp.save_npz(os.path.join(d, "feature_acts.npz"), feature_acts)
    np.save(os.path.join(d, "sae_out.npy"), np.concatenate(outs[k]))
    np.savez(
        os.path.join(d, "sae_params.npz"),
        **{p: getattr(s, p).detach().cpu().numpy() for p in ["W_dec", "b_dec", "W_enc", "b_enc"]},
    )
    print(f"Saved {feature_acts.shape[0]} tokens, {feature_acts.shape[1]} latents, mean L0 {feature_acts.getnnz(axis=1).mean():.1f}, to {d}")
