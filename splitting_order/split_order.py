"""Which children split off first as dictionary width grows? Exact-loss L1 dictionary learning.

Data: two parents (directions v_A, v_B) with mutually exclusive children. A child-k input is
x_k = v_parent + s_k u_k (u_k orthonormal, distinctive component s_k), active with probability pi_k;
otherwise the input is 0 (P(0) > 1/2, so the optimal bias is 0 and we omit it).
Loss (eq. 1): E[ ||x - W z||^2 + 2 lam ||z||_1 ], z >= 0, unit-norm columns of W. Codes are free per
input pattern, so the expected loss is exact. For each width m we take the best of many restarts.
"""
import json, math, sys
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
LAM = 0.05
CHILDREN = [("A1", "A", 0.12, 0.6), ("A2", "A", 0.12, 0.3), ("A3", "A", 0.04, 0.6), ("A4", "A", 0.04, 0.1),
            ("B1", "B", 0.06, 0.9), ("B2", "B", 0.06, 0.15), ("B3", "B", 0.02, 1.0)]
K = len(CHILDREN); D = 2 + K
I = torch.eye(D)
X = torch.stack([I[0 if p == "A" else 1] + s * I[2 + i] for i, (_, p, _, s) in enumerate(CHILDREN)])
P = torch.tensor([pi for _, _, pi, _ in CHILDREN])


def loss_fn(V, Z):
    W = V / V.norm(dim=0, keepdim=True); Zc = Z.clamp(min=0)
    return (P * (((X - Zc @ W.T) ** 2).sum(1) + 2 * LAM * Zc.sum(1))).sum()


def fit(m, seed, steps=5000, lr=5e-3):
    g = torch.Generator().manual_seed(seed)
    if seed % 2 == 0:   # data-based init: columns at random child inputs plus noise
        idx = torch.randint(0, K, (m,), generator=g)
        V = X[idx].T + 0.05 * torch.randn(D, m, generator=g)
    else:
        V = torch.randn(D, m, generator=g)
    Z = 0.3 * torch.rand(K, m, generator=g)
    V.requires_grad_(True); Z.requires_grad_(True)
    opt = torch.optim.Adam([V, Z], lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / steps)))
    for _ in range(steps):
        L = loss_fn(V, Z); opt.zero_grad(); L.backward(); opt.step(); sch.step()
        with torch.no_grad(): Z.clamp_(min=0)
    with torch.no_grad():
        return float(loss_fn(V, Z)), (V / V.norm(dim=0, keepdim=True)).clone(), Z.clamp(min=0).clone()


def split_set(W, Z, tol=1e-4):
    act = Z > tol
    out = []
    for k in range(K):
        if not act[k].any(): continue
        j = int(Z[k].argmax())
        if act[:, j].sum() == 1:     # child k's main latent fires on child k only
            out.append(CHILDREN[k][0])
    return out


def scores(W, Z):
    """Predicted gain from giving child k its own latent, from the unsplit (width-2) solution."""
    res = {}
    for k, (name, _, pi, s) in enumerate(CHILDREN):
        j = int(Z[k].argmax()); w = W[:, j]; xn = float(X[k].norm())
        orth2 = float(X[k] @ X[k] - (X[k] @ w) ** 2)
        r = float((X[k] - W @ Z[k]).norm())
        res[name] = dict(pi=pi, s=s, orth2=round(orth2, 4),
                         move=pi * orth2 * (1 - LAM / xn),          # move the input onto its own aligned latent
                         add=pi * max(r - LAM, 0.0) ** 2)            # eq.-10 gain of a new latent, others frozen
    return res


if __name__ == "__main__":
    widths = [int(a) for a in sys.argv[1].split(",")]; n_seeds = int(sys.argv[2])
    out = {}
    for m in widths:
        best = min((fit(m, sd) for sd in range(n_seeds)), key=lambda t: t[0])
        L, W, Z = best
        groups = []
        for j in range(m):
            members = [CHILDREN[k][0] for k in range(K) if Z[k, j] > 1e-3]
            if members:
                groups.append("{" + ",".join(members) + "}")
        out[m] = dict(loss=L, split=split_set(W, Z), groups=sorted(groups),
                      dead=int(((Z > 1e-4).sum(0) == 0).sum()))
        if m == 2:
            out["scores"] = scores(W, Z)
        print(m, json.dumps(out[m]), flush=True)
    print(json.dumps(out.get("scores", {}), indent=1))
    json.dump(out, open(sys.argv[3], "w"), indent=1)
