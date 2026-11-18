"""Checks of the paper's first-order optimality conditions on a trained SAE's representation.

Notation (matches the paper): z_d is feature d's activity over tokens, A_d the tokens where z_d = 0,
w_d the decoder direction. We work in the rescaled coordinates

    u_d = z_d / mean(z_d)  (>= 0, and exactly 0 where the feature is off)

which is the paper's normalisation shifted by one: z_bar / |min z_bar| = u - 1 whenever min z_d = 0.
The feature-feature condition, eq. (6), projected onto a subset S of the other features, then reads

    1 - cos(w_d, w_S)  in  conv{ u_S[i] : i in A_d }.

Tokens in A_d with every feature in S off all sit at the origin, so (provided one exists) the condition
is equivalent to the *hull gauge*

    g(d, S) = min { sum(y) : y >= 0, sum_i y_i u_S[i] = 1 - cos(w_d, w_S), i in A_d }  <=  1,

a small LP. For |S| = 1 this is the pairwise interval test, g = (1 - cos) / max_{A_d} u_{d'}.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


class FeatureData:
    """Frequent SAE features, ordered by activation frequency (most frequent first)."""

    def __init__(self, feature_acts, W_dec, freq_cutoff=0.005):
        N = feature_acts.shape[0]
        freq = feature_acts.getnnz(axis=0) / N
        idx = np.where(freq > freq_cutoff)[0]
        self.idx = idx[np.argsort(-freq[idx])]  # original SAE feature ids
        self.N, self.F = N, len(self.idx)

        Z = feature_acts[:, self.idx].tocsc().astype(np.float64)
        self.U = (Z @ sp.diags(N / np.asarray(Z.sum(axis=0)).ravel())).tocsc()  # u = z / mean(z)
        self.U_csr = self.U.tocsr()
        B = (self.U > 0).astype(np.float64)
        self.n_active = np.asarray(B.sum(axis=0)).ravel()
        self.coactive = (B.T @ B).toarray()  # tokens on which both features are active

        W = W_dec[self.idx].astype(np.float64)
        self.W_hat = W / np.linalg.norm(W, axis=1, keepdims=True)
        self.cos = self.W_hat @ self.W_hat.T

    def permuted(self, rng):
        """Activity-permutation null: each feature's activity is shuffled independently across tokens.
        Keeps every marginal distribution and the dictionary, destroys all co-activation structure
        (hierarchies, mutual exclusivity, correlations). Feature order and indices are unchanged."""
        new = object.__new__(FeatureData)
        new.__dict__.update(self.__dict__)
        U = self.U.copy()
        for d in range(self.F):
            col = slice(U.indptr[d], U.indptr[d + 1])
            U.indices[col] = rng.choice(self.N, U.indptr[d + 1] - U.indptr[d], replace=False)
        U.has_sorted_indices = False
        U.sort_indices()
        new.U, new.U_csr = U, U.tocsr()
        B = (U > 0).astype(np.float64)
        new.coactive = (B.T @ B).toarray()
        return new

    def target(self, d, S, cos_row=None):
        """The point that must lie in the hull: 1 - cos(w_d, w_S)."""
        return 1 - (self.cos[d] if cos_row is None else cos_row)[S]

    def inactive_points(self, d, S, dedupe=True):
        """Points u_S[i] for tokens i in A_d with at least one feature of S active, as the columns of a
        (|S|, n) matrix, and whether some token in A_d has all of S off (the origin is then a point too).
        Deduplicating shrinks the LP a lot for small S; for large S it costs more than it saves."""
        U_S = self.U[:, S].tocsr()
        d_active = self.U.indices[self.U.indptr[d] : self.U.indptr[d + 1]]
        rows = np.setdiff1d(np.unique(U_S.nonzero()[0]), d_active, assume_unique=True)
        origin_present = self.N - self.n_active[d] > len(rows)
        if dedupe:
            P = U_S[rows].toarray()
            return (np.unique(P, axis=0).T if len(P) else P.T), origin_present
        return U_S[rows].T.tocsc(), origin_present


def hull_gauge(fd, d, S, cos_row=None, method="highs"):
    """g(d, S) from the module docstring; the condition holds iff g <= 1. np.inf means the target is
    outside the cone of the inactive points altogether (e.g. a feature in S never fires while d is off:
    a strict hierarchy). cos_row replaces d's cosines (for the shuffled-cosine null). For |S| in the
    hundreds or more, method="highs-ipm" (interior point) is several times faster than the default."""
    S = np.asarray(S)
    t = fd.target(d, S, cos_row)
    P, origin_present = fd.inactive_points(d, S, dedupe=len(S) <= 200)
    n = P.shape[1]
    if n == 0:
        return 0.0 if np.allclose(t, 0) else np.inf
    A_eq, b_eq = P, t
    if not origin_present:  # every A_d token has some S-feature on, so the weights must sum to exactly 1
        A_eq, b_eq = sp.vstack([sp.csc_matrix(P), np.ones((1, n))]), np.append(t, 1.0)
    res = linprog(np.ones(n), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method=method)
    if res.status == 2:  # infeasible
        return np.inf
    assert res.status == 0, res.message
    if not origin_present:  # feasibility is all that's left to test; report 1 for "holds"
        return 1.0
    return res.fun


def pairwise_max_inactive(fd, top_k=64):
    """M[d, d'] = max over A_d of u_{d'}: the upper end of the pairwise interval.
    For each d' we only scan its top_k largest activations; the max over A_d is the first of those at
    which d is off. Entries where d is on for all top_k are filled exactly afterwards."""
    M = np.full((fd.F, fd.F), np.nan)
    for dp in range(fd.F):
        col = slice(fd.U.indptr[dp], fd.U.indptr[dp + 1])
        rows, vals = fd.U.indices[col], fd.U.data[col]
        order = np.argsort(-vals)[:top_k]
        rows, vals = rows[order], vals[order]
        on = fd.U_csr[rows].toarray() > 0  # (top_k, F): which features are on at d''s biggest tokens
        first_off = np.argmax(~on, axis=0)
        found = ~on[first_off, np.arange(fd.F)]
        M[found, dp] = vals[first_off[found]]
        # d on at all of d''s active tokens: max over A_d is 0 (d' lives entirely inside d)
        M[fd.coactive[:, dp] == fd.n_active[dp], dp] = 0.0
    for d, dp in zip(*np.where(np.isnan(M))):
        z_d = fd.U[:, d].toarray().ravel()
        u_dp = fd.U[:, dp].toarray().ravel()
        M[d, dp] = u_dp[z_d == 0].max()
    np.fill_diagonal(M, np.nan)
    return M


def pairwise_check(fd, M=None, tol=1e-3):
    """Corrected, vectorised version of the original pairwise test, over all ordered pairs (d, d').
    Returns the pairwise gauge (1 - cos) / M, a boolean violation matrix, and M.
    Two ways to violate (both flagged in `violations`):
      upper end  1 - cos > M: d' doesn't get big enough while d is off (strict hierarchy: M = 0, gauge inf)
      lower end  1 - cos < min over A_d of u_{d'}: only possible if d' is on at every token where d is off,
                 i.e. the two features are never off together (the paper's Fig 1B case)"""
    if M is None:
        M = pairwise_max_inactive(fd)
    t = 1 - fd.cos
    with np.errstate(divide="ignore", invalid="ignore"):
        gauge = np.where(t <= tol, 0.0, t / M)
    np.fill_diagonal(gauge, np.nan)
    violations = gauge > 1 + tol
    both_off = fd.N - fd.n_active[:, None] - fd.n_active[None, :] + fd.coactive
    np.fill_diagonal(both_off, 1)
    for d, dp in zip(*np.where(both_off == 0)):
        z_d = fd.U[:, d].toarray().ravel()
        min_inactive = fd.U[:, dp].toarray().ravel()[z_d == 0].min()
        violations[d, dp] |= t[d, dp] < min_inactive - tol
    return gauge, violations, M


def choose_subset(fd, d, k, strategy, rng, pair_gauge=None):
    """Which k other features to test against feature d.
    random:      uniform over frequent features
    top_cos:     largest |cos(w_d, w_d')|, the features d is most geometrically entangled with
    subordinate: largest P(d on | d' on), candidates for d' being a sub-feature of d (hierarchies)
    worst_pairs: largest pairwise gauge, i.e. the pairs closest to violating; checks whether nearly
                 binding pairwise constraints combine into a violated joint one"""
    others = np.delete(np.arange(fd.F), d)
    if strategy == "random":
        return rng.choice(others, k, replace=False)
    if strategy == "top_cos":
        score = np.abs(fd.cos[d, others])
    elif strategy == "subordinate":
        score = fd.coactive[d, others] / fd.n_active[others]
    elif strategy == "worst_pairs":
        score = np.nan_to_num(pair_gauge[d, others], nan=-np.inf)
    return others[np.argsort(-score)[:k]]


def residual_kkt(fd, E, chunk=256):
    """Direct test of the general first-order (LASSO-KKT) condition, eq. (48): at an optimum
    r_d[i] = w_hat_d . e[i] equals lambda on d's active set and is <= lambda where d is off.
    E is the (N, d_in) reconstruction residual. Returns per-feature summary statistics."""
    out = {k: np.zeros(fd.F) for k in ["lam_hat", "active_iqr", "frac_inactive_above", "inactive_sd"]}
    for start in range(0, fd.F, chunk):
        feats = np.arange(start, min(start + chunk, fd.F))
        R = E @ fd.W_hat[feats].T  # (N, chunk)
        for j, d in enumerate(feats):
            on = np.zeros(fd.N, bool)
            on[fd.U.indices[fd.U.indptr[d] : fd.U.indptr[d + 1]]] = True
            r_on, r_off = R[on, j], R[~on, j]
            lam = np.median(r_on)
            q25, q75 = np.percentile(r_on, [25, 75])
            out["lam_hat"][d] = lam
            out["active_iqr"][d] = q75 - q25
            out["frac_inactive_above"][d] = np.mean(r_off > lam)
            out["inactive_sd"][d] = r_off.std()
    return out
