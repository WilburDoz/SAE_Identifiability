"""Extension of exact_localmin.py: non-orthogonal features, superposition, variable magnitudes.

Exact expected loss over every activation pattern (magnitudes discretised to levels, so the
enumeration stays exact). Start at the ground-truth (un-absorbed) dictionary + 1e-3 noise and
run descent. Compare hierarchical data with an independent control of the same frequencies.

geom:  orth      orthonormal features
       pc+ / pc- random interference (noise 0.2) and every parent-child cosine set to +0.3 / -0.3
       superpos  n random unit features in n-1 dimensions
mags:  fixed     all active magnitudes 1
       var       active magnitudes uniform on {0.5, 1.0, 1.5}, independent across features
Single process, single thread.
"""
import itertools, json, math, sys
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def spec(name):
    if name.startswith("c4"):
        n, par, kids = 4, 0, [1]
        def pr(b):
            p = (.25 if b[0] else .75) * math.prod(.05 if x else .95 for x in b[2:])
            if name == "c4_indep":
                return p * (.05 if b[1] else .95)
            return p * ((.2 if b[1] else .8) if b[0] else (0. if b[1] else 1.))
    elif name.startswith("c12"):
        n, par, kids = 12, 0, [1]
        def pr(b):
            p = (.4 if b[0] else .6) * math.prod(.15 if x else .85 for x in b[2:])
            if name == "c12_indep":
                return p * (.24 if b[1] else .76)
            return p * ((.6 if b[1] else .4) if b[0] else (0. if b[1] else 1.))
    elif name.startswith("tang"):
        n, par, kids = 6, 0, [1, 2, 3]
        q = 1 - .95 ** 3
        def pr(b):
            p = math.prod(.05 if x else .95 for x in b[1:])
            if name == "tang_indep":
                return p * (q if b[0] else 1 - q)
            return p if b[0] == any(b[1:4]) else 0.
    return n, par, kids, pr


def acts(name, mags):
    n, par, kids, pr = spec(name)
    levels = [1.0] if mags == "fixed" else [0.5, 1.0, 1.5]
    rows, ps = [], []
    for b in itertools.product([0, 1], repeat=n):
        p = pr(b)
        if p <= 0 or not any(b):
            continue
        on = [i for i in range(n) if b[i]]
        for lv in itertools.product(levels, repeat=len(on)):
            a = [0.0] * n
            for i, v in zip(on, lv):
                a[i] = v
            rows.append(a); ps.append(p / len(levels) ** len(on))
    return torch.tensor(rows), torch.tensor(ps), n, par, kids


def geometry(n, par, kids, geom, gseed=7):
    if geom == "orth":
        return torch.eye(n)
    g = torch.Generator().manual_seed(gseed)
    if geom == "superpos":
        F = torch.randn(n, n - 1, generator=g)
    else:
        F = torch.eye(n) + 0.2 * torch.randn(n, n, generator=g)
    F = F / F.norm(dim=1, keepdim=True)
    if geom in ("pc+", "pc-"):
        c = 0.3 if geom == "pc+" else -0.3
        f0 = F[par].clone()
        for k in kids:
            u = F[k] - (F[k] @ f0) * f0
            F[k] = math.sqrt(1 - c * c) * u / u.norm() + c * f0
    return F                                               # n x D, unit rows


def topk_mask(pre, k):
    v, i = pre.topk(k, dim=1)
    return torch.zeros_like(pre).scatter(1, i, torch.relu(v))


class Model:
    def __init__(self, kind, F, A, lam, k, eps, seed):
        n, D = F.shape
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: eps * torch.randn(*s, generator=g)
        self.kind, self.lam, self.k = kind, lam, k
        self.V = F.T.clone() + r(D, n)                     # decoder = V / |V| (normalised in the loss,
        if kind == "dl":                                   #  so Adam needs no projection step)
            self.Z = torch.relu(A - lam)
            self.params = [self.V, self.Z]
        else:
            self.We = torch.linalg.pinv(F.T) + r(n, D)     # dual basis (pseudo-inverse if superposed)
            self.be = torch.full((n,), -lam if kind == "sae" else 0.)
            self.params = [self.We, self.be, self.V]
        for p in self.params:
            p.requires_grad_(True)
        self.project()

    @property
    def Wd(self):
        return self.V / self.V.norm(dim=0, keepdim=True)

    def project(self):
        with torch.no_grad():
            if self.kind == "dl":
                self.Z.clamp_(min=0)

    def codes(self, X):
        if self.kind == "dl":
            return self.Z
        pre = X @ self.We.T + self.be
        return torch.relu(pre) if self.kind == "sae" else topk_mask(pre, self.k)

    def loss(self, X, P):
        Z = self.codes(X)
        per = ((X - Z @ self.Wd.T) ** 2).sum(1)
        if self.kind != "topk":
            per = per + 2 * self.lam * Z.sum(1)
        return (P * per).sum()


def metrics(m, F, A, X, P, par, kids):
    with torch.no_grad():
        Z = m.codes(X)
        C = F @ (m.Wd / m.Wd.norm(dim=0))                  # cos(true feature i, decoder j)
        child_lat = [int(C[c].argmax()) for c in kids]
        absorb = sum(float(C[par, j]) for j in child_lat) / len(kids)
        others = [j for j in range(C.shape[1]) if j not in child_lat]
        Pl = others[int(C[par, others].argmax())]
        kid_on = A[:, kids].sum(1) > 0
        w = (A[:, par] > 0) & kid_on
        a = (A[:, par] > 0) & ~kid_on
        fires = (Z[:, Pl] > 1e-6).double()
        rec_w = float((P[w] * fires[w]).sum() / P[w].sum())
        rec_a = float((P[a] * fires[a]).sum() / P[a].sum()) if a.any() else float("nan")
    return absorb, rec_w, rec_a


def run(kind, data, geom, mags, lam=0.05, k=None, eps=1e-3, seed=0, steps=5000, lr=2e-3):
    A, P, n, par, kids = acts(data, mags)
    F = geometry(n, par, kids, geom)
    X = A @ F
    gt = Model(kind, F, A, lam, k, 0.0, 0)
    with torch.no_grad():
        L_gt = float(gt.loss(X, P))
    m = Model(kind, F, A, lam, k, eps, seed)
    opt = torch.optim.Adam(m.params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / steps)))
    for t in range(steps):
        L = m.loss(X, P)
        opt.zero_grad(); L.backward(); opt.step(); sched.step(); m.project()
    with torch.no_grad():
        L_end = float(m.loss(X, P))
    g_abs, g_rw, g_ra = metrics(gt, F, A, X, P, par, kids)
    e_abs, e_rw, e_ra = metrics(m, F, A, X, P, par, kids)
    return dict(model=kind + (f"-k{k}" if k else ""), data=data, geom=geom, mags=mags,
                pc_cos=float(F[par] @ F[kids[0]]), npat=len(P), loss_gt=L_gt, loss_end=L_end,
                gt_absorb=g_abs, end_absorb=e_abs, gt_par_w_child=g_rw, end_par_w_child=e_rw,
                end_par_alone=e_ra)


if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    for c in cfgs:
        r = run(**c)
        print(json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}), flush=True)
