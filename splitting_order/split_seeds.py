"""Seed-level view of the splitting experiments.

For every restart (not just the best), record the loss, the grouping of children by latent, which parent
received the extra latent, and whether the solution violates eq. (6) in the nested-support sense:
a latent whose set of active inputs is a strict subset of another latent's, with non-aligned decoders.
"""
import sys, json, random, torch
sys.path.insert(0, ".")
import split_order as S

torch.set_num_threads(1)


def describe(W, Z, names, parent_of):
    m = W.shape[1]
    supp = [frozenset(k for k in range(len(names)) if Z[k, j] > 1e-3) for j in range(m)]
    groups = sorted("{" + ",".join(names[k] for k in sorted(s)) + "}" for s in supp if s)
    viol = 0
    for a in range(m):
        for b in range(m):
            if a != b and supp[a] and supp[a] < supp[b] and float(W[:, a] @ W[:, b]) < 0.99:
                viol += 1
    n_lat = {p: sum(1 for s in supp if any(parent_of[k] == p for k in s)) for p in ["A", "B"]}
    return groups, viol, n_lat


def set_instance(kids):
    S.CHILDREN = kids; S.K = len(kids); S.D = 2 + S.K
    I = torch.eye(S.D)
    S.X = torch.stack([I[0 if p == "A" else 1] + s * I[2 + i] for i, (_, p, _, s) in enumerate(kids)])
    S.P = torch.tensor([k[2] for k in kids])


out = {"example": {}, "random": []}

# ---- Experiment 1 instance, widths 3-6, 20 seeds each
kids1 = list(S.CHILDREN); set_instance(kids1)
names = [k[0] for k in kids1]; parent_of = [k[1] for k in kids1]
for m in [3, 4, 5, 6]:
    runs = []
    for sd in range(20):
        L, W, Z = S.fit(m, sd)
        g, v, n = describe(W, Z, names, parent_of)
        runs.append(dict(seed=sd, loss=L, groups=g, violations=v))
    out["example"][m] = runs
    best = min(r["loss"] for r in runs)
    structs = {}
    for r in runs:
        key = " ".join(r["groups"])
        structs.setdefault(key, []).append(r["loss"])
    print(f"example width {m}: best {best:.4f}, eq6 violations in {sum(r['violations'] > 0 for r in runs)}/20 runs", flush=True)
    for key, ls in sorted(structs.items(), key=lambda kv: min(kv[1])):
        print(f"   {len(ls):2d} runs  loss {min(ls):.4f}-{max(ls):.4f}  {key}", flush=True)

# ---- Same 20 random instances as split_first.py, width 3, 10 seeds each
rng = random.Random(0)
for inst in range(20):
    kids = []
    for p in ["A", "B"]:
        for c in range(rng.randint(2, 4)):
            kids.append((f"{p}{c+1}", p, rng.uniform(0.02, 0.10), rng.uniform(0.1, 1.0)))
    tot = sum(k[2] for k in kids)
    kids = [(n, p, pi * min(1.0, 0.45 / tot), s) for n, p, pi, s in kids]
    set_instance(kids)
    names = [k[0] for k in kids]; parent_of = [k[1] for k in kids]
    L2, W2, Z2 = min((S.fit(2, sd, steps=4000) for sd in range(6)), key=lambda t: t[0])
    spread = {}
    for p in ["A", "B"]:
        spread[p] = 0.0
        for k in range(S.K):
            if kids[k][1] == p:
                w = W2[:, int(Z2[k].argmax())]
                spread[p] += kids[k][2] * float(S.X[k] @ S.X[k] - (S.X[k] @ w) ** 2)
    pred = max(spread, key=spread.get)
    runs = []
    for sd in range(10):
        L, W, Z = S.fit(3, sd, steps=4000)
        g, v, n = describe(W, Z, names, parent_of)
        who = "A" if n["A"] > n["B"] else ("B" if n["B"] > n["A"] else "tie")
        runs.append(dict(seed=sd, loss=L, split=who, violations=v))
    best = min(runs, key=lambda r: r["loss"])
    alt = [r["loss"] for r in runs if r["split"] not in (best["split"], "tie")]
    row = dict(inst=inst, pred=pred, spread=spread, best_split=best["split"], best_loss=best["loss"],
               frac_runs_pred=sum(r["split"] == pred for r in runs) / len(runs),
               alt_best_loss=min(alt) if alt else None,
               violations=sum(r["violations"] > 0 for r in runs), runs=runs)
    out["random"].append(row)
    print(f"inst {inst:2d}: predicted {pred}, best-of-10 split {best['split']}, runs splitting predicted parent "
          f"{row['frac_runs_pred']:.1f}, best other-parent loss {row['alt_best_loss']}, "
          f"best {best['loss']:.4f}, eq6 violations {row['violations']}/10", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
