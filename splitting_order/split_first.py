"""Across random two-parent instances: which parent receives the first extra latent (width 2 -> 3)?

Candidate predictors, computed from the width-2 optimum (one latent per parent):
  spread_p = sum_{children k of p} pi_k * orth_k^2   (frequency-weighted squared distance from the parent latent)
  freq_p   = sum pi_k                                (total frequency)
  nkids_p  = number of children
"""
import sys, json, random, torch
sys.path.insert(0, ".")
import split_order as S

torch.set_num_threads(1)
N_INST, OUT = int(sys.argv[1]), sys.argv[2]
rng = random.Random(0)
results = []
for inst in range(N_INST):
    kids = []
    for p in ["A", "B"]:
        for c in range(rng.randint(2, 4)):
            kids.append((f"{p}{c+1}", p, rng.uniform(0.02, 0.10), rng.uniform(0.1, 1.0)))
    tot = sum(k[2] for k in kids)
    kids = [(n, p, pi * min(1.0, 0.45 / tot), s) for n, p, pi, s in kids]   # keep P(0) > 1/2
    S.CHILDREN = kids; S.K = len(kids); S.D = 2 + S.K
    I = torch.eye(S.D)
    S.X = torch.stack([I[0 if p == "A" else 1] + s * I[2 + i] for i, (_, p, _, s) in enumerate(kids)])
    S.P = torch.tensor([k[2] for k in kids])
    L2, W2, Z2 = min((S.fit(2, sd, steps=4000) for sd in range(6)), key=lambda t: t[0])
    pred = {}
    for p in ["A", "B"]:
        ks = [k for k in range(S.K) if kids[k][1] == p]
        spread = 0.0
        for k in ks:
            j = int(Z2[k].argmax()); w = W2[:, j]
            spread += kids[k][2] * float(S.X[k] @ S.X[k] - (S.X[k] @ w) ** 2)
        pred[p] = dict(spread=spread, freq=sum(kids[k][2] for k in ks), nkids=len(ks))
    L3, W3, Z3 = min((S.fit(3, sd, steps=4000) for sd in range(10)), key=lambda t: t[0])
    n_lat = {}
    for p in ["A", "B"]:
        ks = [k for k in range(S.K) if kids[k][1] == p]
        n_lat[p] = int(sum(bool((Z3[ks, j] > 1e-3).any()) for j in range(3)))
    winner = "A" if n_lat["A"] > n_lat["B"] else ("B" if n_lat["B"] > n_lat["A"] else "tie")
    row = dict(inst=inst, winner=winner, n_lat=n_lat, pred=pred, L2=L2, L3=L3,
               by_spread=max(pred, key=lambda q: pred[q]["spread"]),
               by_freq=max(pred, key=lambda q: pred[q]["freq"]),
               by_nkids=max(pred, key=lambda q: pred[q]["nkids"]) if pred["A"]["nkids"] != pred["B"]["nkids"] else "tie")
    results.append(row)
    print(json.dumps(row), flush=True)
json.dump(results, open(OUT, "w"), indent=1)
ok = [r for r in results if r["winner"] != "tie"]
for key in ["by_spread", "by_freq", "by_nkids"]:
    valid = [r for r in ok if r[key] != "tie"]
    print(key, f"{sum(r[key] == r['winner'] for r in valid)}/{len(valid)} correct")
