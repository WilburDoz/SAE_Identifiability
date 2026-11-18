"""Runs every setting of the toy local-minimum experiment (App. G.2) and writes results.jsonl.

104 runs: {L1 dictionary learning, L1 SAE, TopK k=1} x {Chanin 4-feature toy, Tang et al. Def. 3.2 toy}
x {hierarchical, independent control} x {orthogonal, parent-child cos +0.3, -0.3, superposition}
x {fixed, variable magnitudes}, plus TopK k=2 on the 12-feature toy for each geometry.
Single process, single thread.

Usage (from this folder):  OMP_NUM_THREADS=1 python run_toy_localmin.py > results.jsonl
"""
import json
import subprocess
import sys

C = []
for geom in ["orth", "pc+", "pc-", "superpos"]:
    for mags in ["fixed", "var"]:
        for d in ["c4_indep", "c4_hier", "tang_indep", "tang_hier"]:
            C.append(dict(kind="dl", data=d, geom=geom, mags=mags, steps=6000))
            C.append(dict(kind="sae", data=d, geom=geom, mags=mags, steps=6000))
            C.append(dict(kind="topk", data=d, geom=geom, mags=mags, k=1, steps=6000))
for geom in ["orth", "pc+", "pc-", "superpos"]:
    for d in ["c12_indep", "c12_hier"]:
        C.append(dict(kind="topk", data=d, geom=geom, mags="fixed", k=2, steps=6000))

subprocess.run([sys.executable, "-W", "ignore", "exact_localmin.py", json.dumps(C)], check=True)
