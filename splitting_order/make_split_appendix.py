"""Splitting-order appendix: figure, tables and LaTeX text, generated from saved results.

Inputs (same folder): split_sweep.json (split_order.py), split_first.json (split_first.py),
split_seeds.json (split_seeds.py). Recomputes the best-division predictions (pure numpy, instant).
Outputs: fig_split_order.pdf/.png, appendix_split_order.tex   (to $OUT_DIR if set, else this folder)
"""
import itertools, json, math, os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", HERE)
LAM = 0.05
BLUE, ORANGE, INK, MUTED, RULE, BAR = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#cfcdc8", "#8a8984"
load = lambda f: json.load(open(os.path.join(HERE, f)))
sweep, first, seeds = load("split_sweep.json"), load("split_first.json"), load("split_seeds.json")

EXAMPLE = [("A1", "A", 0.12, 0.6), ("A2", "A", 0.12, 0.3), ("A3", "A", 0.04, 0.6), ("A4", "A", 0.04, 0.1),
           ("B1", "B", 0.06, 0.9), ("B2", "B", 0.06, 0.15), ("B3", "B", 0.02, 1.0)]


def inputs(kids):
    D = 2 + len(kids); I = np.eye(D)
    X = np.stack([I[0 if p == "A" else 1] + s * I[2 + i] for i, (_, p, _, s) in enumerate(kids)])
    return X, np.array([k[2] for k in kids])


def captured(idx, X, P, lam):
    """max_w sum_k pi_k (<x_k,w> - lam)_+^2 over unit w; lam = 0 gives the top eigenvalue of C_G."""
    Xg, Pg = X[idx], P[idx]
    C = (Xg * Pg[:, None]).T @ Xg
    w = np.linalg.eigh(C)[1][:, -1]; w *= np.sign(w @ Xg.sum(0))
    if lam == 0:
        return float(w @ C @ w)
    for _ in range(500):
        a = np.maximum(Xg @ w - lam, 0)
        w_new = (Pg * a) @ Xg; w_new /= np.linalg.norm(w_new)
        if np.allclose(w_new, w): break
        w = w_new
    return float(Pg @ np.maximum(Xg @ w - lam, 0) ** 2)


def best_division(idx, X, P, lam):
    idx = list(idx); base = captured(idx, X, P, lam); best = (0.0, None)
    for r in range(1, len(idx)):
        for g1 in itertools.combinations(idx, r):
            if idx[0] not in g1: continue
            g2 = [k for k in idx if k not in g1]
            gain = captured(list(g1), X, P, lam) + captured(g2, X, P, lam) - base
            if gain > best[0] + 1e-12: best = (gain, (list(g1), g2))
    return best


# ---------------------------------------------------------------- 20 random instances
rng = random.Random(0)
inst_rows = []
for inst in range(20):
    kids = []
    for p in ["A", "B"]:
        for c in range(rng.randint(2, 4)):
            kids.append((f"{p}{c+1}", p, rng.uniform(0.02, 0.10), rng.uniform(0.1, 1.0)))
    tot = sum(k[2] for k in kids)
    kids = [(n, p, pi * min(1.0, 0.45 / tot), s) for n, p, pi, s in kids]
    X, P = inputs(kids)
    gains = {lam: {p: best_division([k for k in range(len(kids)) if kids[k][1] == p], X, P, lam)[0] for p in "AB"}
             for lam in [0, LAM]}
    r = first[inst]
    assert r["inst"] == inst
    inst_rows.append(dict(winner=r["winner"], spread=r["pred"], gains=gains,
                          pred_div={lam: max(g, key=g.get) for lam, g in gains.items()},
                          by_spread=r["by_spread"], by_freq=r["by_freq"], by_nkids=r["by_nkids"]))
n = len(inst_rows)
acc = {
    "div0": sum(r["pred_div"][0] == r["winner"] for r in inst_rows),
    "divlam": sum(r["pred_div"][LAM] == r["winner"] for r in inst_rows),
    "spread": sum(r["by_spread"] == r["winner"] for r in inst_rows),
    "freq": sum(r["by_freq"] == r["winner"] for r in inst_rows),
}
kid_valid = [r for r in inst_rows if r["by_nkids"] != "tie"]
acc["nkids"] = sum(r["by_nkids"] == r["winner"] for r in kid_valid); acc["nkids_n"] = len(kid_valid)
spread_miss_ratio = max(max(r["spread"]["A"]["spread"], r["spread"]["B"]["spread"]) /
                        min(r["spread"]["A"]["spread"], r["spread"]["B"]["spread"])
                        for r in inst_rows if r["by_spread"] != r["winner"])

# ---------------------------------------------------------------- seeds / local optima
R = seeds["random"]
frac_best = [sum(run["split"] == r["best_split"] for run in r["runs"]) / len(r["runs"]) for r in R]
alt = [(r["alt_best_loss"] - r["best_loss"]) / r["best_loss"] for r in R if r["alt_best_loss"] is not None]
viol_runs = sum(run["violations"] > 0 for r in R for run in r["runs"]) + \
            sum(run["violations"] > 0 for runs in seeds["example"].values() for run in runs)
n_runs = sum(len(r["runs"]) for r in R) + sum(len(runs) for runs in seeds["example"].values())
viol_gaps = []
for r in R:
    for run in r["runs"]:
        if run["violations"]:
            viol_gaps.append((run["loss"] - r["best_loss"]) / r["best_loss"])
for m, runs in seeds["example"].items():
    best = min(x["loss"] for x in runs)
    viol_gaps += [(x["loss"] - best) / best for x in runs if x["violations"]]

# ---------------------------------------------------------------- example: greedy chain
X, P = inputs(EXAMPLE); names = [k[0] for k in EXAMPLE]
groups = [[0, 1, 2, 3], [4, 5, 6]]
chain = {}
for m in range(3, 8):
    cand = [(best_division(g, X, P, LAM), i) for i, g in enumerate(groups) if len(g) > 1]
    (gain, (g1, g2)), i = max(cand, key=lambda c: c[0][0])
    groups = groups[:i] + [g1, g2] + groups[i + 1:]
    chain[m] = sorted(groups, key=lambda g: (names[g[0]][0], g))

fmt = lambda gs: " ".join("\\{" + ",".join(gs_) + "\\}" for gs_ in gs)
fmt_found = lambda strs: " ".join(s.replace("{", "\\{").replace("}", "\\}") for s in strs)

# ---------------------------------------------------------------- figure
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.7), constrained_layout=True)
ax = axs[0]
ax.axhline(0, color=RULE, lw=0.8, zorder=0); ax.axvline(0, color=RULE, lw=0.8, zorder=0)
for who, col, mk in [("A", BLUE, "o"), ("B", ORANGE, "s")]:
    pts = [r for r in inst_rows if r["winner"] == who]
    xs = [math.log(r["spread"]["A"]["spread"] / r["spread"]["B"]["spread"]) for r in pts]
    ys = [math.log(r["gains"][LAM]["A"] / r["gains"][LAM]["B"]) for r in pts]
    ax.scatter(xs, ys, s=26, c=col, marker=mk, edgecolors="white", linewidths=0.8, zorder=3,
               label=f"parent {who} split first")
ax.set_xlabel("log(spread A / spread B)")
ax.set_ylabel("log(best-division gain A / B)")
ax.set_title("A)  Predicting the first split", loc="left", fontweight="bold", pad=6)
ax.legend(frameon=False, fontsize=7, loc="upper left", handletextpad=0.2)
ax = axs[1]
order = sorted(range(len(frac_best)), key=lambda i: -frac_best[i])
ax.bar(range(len(order)), [frac_best[i] for i in order], color=BAR, width=0.8)
ax.axhline(np.mean(frac_best), color=INK, lw=0.8, ls="--")
ax.text(len(order) - 0.5, np.mean(frac_best) + 0.03, f"mean {np.mean(frac_best):.2f}", ha="right", fontsize=7, color=INK)
ax.set_ylim(0, 1.05); ax.set_xticks([])
ax.set_xlabel("instance (sorted)")
ax.set_ylabel("fraction of runs reaching\nthe best choice of parent")
ax.set_title("B)  Individual training runs", loc="left", fontweight="bold", pad=6)
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(OUT, f"fig_split_order.{ext}"), dpi=300)

# ---------------------------------------------------------------- tables and text
t_children = "\n".join(f"{nm} & {p} & {pi:.2f} & {s:.2f} \\\\" for nm, p, pi, s in EXAMPLE)
rows = []
for m in range(2, 9):
    found = sweep[str(m)]
    pred = "--" if m == 2 or m == 8 else fmt([[names[k] for k in g] for g in chain[m]])
    if m == 2:
        pred = "(start)"
    if m == 8:
        pred = "(no further split)"
    rows.append(f"{m} & {fmt_found(found['groups'])} & {found['loss']:.4f} & {pred} \\\\")
t_sweep = "\n".join(rows)

TEX = r"""% ---- generated by make_split_appendix.py; refs use labels from the main paper; add \label{app:wide_dictionary} to App. F ----
\section{Which parents split first? Splitting order at finite width}
\label{app:split-order}

Eq.~\eqref{eq:feature_feature_constraint} constrains the \emph{form} of splitting. At any width, a parent latent cannot coexist with a latent that fires only on one of its children (unless the two point in the same direction), so each parent's inputs are shared among latents that each cover a group of its children. In the wide limit every child receives its own latent (Section~\ref{sec:feature_splitting_limit}). Which groups form at intermediate widths is not determined by these necessary conditions. Here we derive a simple rule for which latent splits next, and test it in toy models where the expected loss can be computed exactly.

\paragraph{Setting.} Two parents $p\in\{A,B\}$ have orthonormal directions $v_A,v_B$, and each has mutually exclusive children $k$ with orthonormal child-specific directions $u_k$. A child-$k$ input is $x_k=v_{p(k)}+s_k u_k$, occurring with probability $\pi_k$, where $s_k$ measures how distinct the child is from its parent; otherwise the input is $0$. Because more than half of the inputs are zero, the geometric median, which sets the bias in the wide limit (App.~\ref{app:wide_dictionary}), is zero, and we omit the bias. We minimise eq.~\eqref{eq:dictionary_learning_problem} with $\lambda=0.05$, unit-norm dictionary columns and codes optimised separately for each input. With finitely many distinct inputs, the expected loss is a finite weighted sum and is computed exactly. At each width we keep the best of several random restarts (Adam, learning rate $5\times10^{-3}$ with cosine decay, 4000--5000 steps).

\paragraph{A rule for which latent splits.} Suppose each input is coded by a single latent, and a latent with unit direction $w$ covers a group $G$ of inputs. Its optimal codes are $z_k=\langle x_k,w\rangle-\lambda$, and substituting them gives a per-input loss of $\|x_k\|^2-(\langle x_k,w\rangle-\lambda)^2$. The group's loss is therefore
\begin{equation}
L(G)=\sum_{k\in G}\pi_k\|x_k\|^2-\max_{\|w\|=1}\sum_{k\in G}\pi_k\big(\langle x_k,w\rangle-\lambda\big)_+^2 .
\end{equation}
For small $\lambda$ the maximisation is solved by the top eigenvector of $C_G=\sum_{k\in G}\pi_k x_kx_k^\top$, so $L(G)\approx\sum_{k\in G}\pi_k\|x_k\|^2-\lambda_{\max}(C_G)$. Dividing $G$ into $G_1$ and $G_2$ then reduces the loss by
\begin{equation}
\Delta(G_1,G_2)=\lambda_{\max}(C_{G_1})+\lambda_{\max}(C_{G_2})-\lambda_{\max}(C_G)\;\ge 0 ,
\end{equation}
which is non-negative because $\lambda_{\max}$ is subadditive. We predict that the next latent goes to the group whose best division has the largest gain. For two children $a$ and $b$ at angle $\theta_{ab}$, the gain is the smaller eigenvalue of their $2\times2$ weighted Gram matrix:
\begin{equation}
\Delta\;\ge\;\frac{\pi_a\pi_b\|x_a\|^2\|x_b\|^2\sin^2\theta_{ab}}{\pi_a\|x_a\|^2+\pi_b\|x_b\|^2},
\end{equation}
with near-equality when the gain is small. Splitting therefore pays when both children are frequent and point in different directions. The number of children matters only through these gains.

\paragraph{A detailed example.} Table~\ref{tab:split-children} lists seven children of two parents, and Table~\ref{tab:split-sweep} shows the best solution found at each width (best of 10 restarts) alongside the greedy prediction obtained by applying the rule repeatedly from width 2. Three features stand out. Splits divide a parent's children into groups rather than isolating single children. The least distinctive children (A4 and B2) are coded by two sibling latents until they receive their own latent last. And the greedy prediction matches the best solution at widths 3, 4, 6 and 7, up to these shared children, but at width 5 it splits the other parent: the best solutions at successive widths need not refine one another.

\paragraph{Which parent splits first.} We generated %%N%% random instances, each with two parents of two to four children, frequencies $\pi_k\sim\mathcal U(0.02,0.10)$ (rescaled so that $\sum_k\pi_k\le0.45$) and distinctiveness $s_k\sim\mathcal U(0.1,1.0)$. For each we found the best solution at width 2 (best of 6 restarts) and width 3 (best of 10), and recorded which parent received the extra latent. The best-division rule predicted this correctly in all %%N%% instances, both with the small-$\lambda$ eigenvalue formula and with $\lambda=0.05$ (Table~\ref{tab:split-predictors}, Fig.~\ref{fig:split-order}A). Simpler predictors computed from the width-2 solution did worse. The frequency-weighted spread of each parent's children about its latent, $\sum_k\pi_kd_k^2$, was right in %%SPREAD%% instances, and its errors were all close calls (spreads within a factor of %%MISSRATIO%%). Total child frequency was right in %%FREQ%%, and number of children in %%NKIDS%% of the %%NKIDSN%% instances without a tie.

\paragraph{Local optima and eq.~\eqref{eq:feature_feature_constraint}.} Individual training runs often stop at worse local optima. Across 10 restarts per instance, a run reached the best solution's choice of parent in %%FRACMEAN%% of runs on average (as low as %%FRACMIN%% in one instance; Fig.~\ref{fig:split-order}B). In %%NALT%% of %%N%% instances, at least one run split the other parent, at a median loss %%ALTMED%% higher than the best solution. Eq.~\eqref{eq:feature_feature_constraint}, by contrast, held in %%NOVIOL%% of %%NRUNS%% runs: no latent fired only on a strict subset of another latent's inputs while pointing in a different direction. The %%NVIOL%% exceptions were all worse than the best solution at their width (by %%VMIN%%--%%VMAX%%), consistent with eq.~\eqref{eq:feature_feature_constraint} being necessary for local optimality.

\paragraph{Limits.} The rule predicts the first split reliably, but it is greedy. The best solutions at successive widths need not refine one another, as in $k$-means clustering, and the least distinctive children are coded by combinations of latents, violating the single-latent assumption. Finding the global optimum at a given width is a combinatorial clustering problem, and gradient training frequently stops at worse local optima. These experiments use L1 dictionary learning with mutually exclusive children of fixed magnitude.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{fig_split_order.pdf}
\caption{\textbf{A)} For each of the %%N%% random instances, the log ratio of the two parents' frequency-weighted spreads against the log ratio of their best-division gains, coloured by which parent received the extra latent in the best width-3 solution. The best-division rule (sign of the vertical axis) is correct in every instance; the spread rule (sign of the horizontal axis) fails in %%SPREADMISS%% close calls. \textbf{B)} Fraction of the 10 training runs per instance that split the same parent as the best solution, sorted; the dashed line is the mean.}
\label{fig:split-order}
\end{figure}

\begin{table}[t]
\centering
\caption{Children in the detailed example: parent, frequency $\pi_k$ and distinctiveness $s_k$.}
\label{tab:split-children}
\small
\begin{tabular}{llcc}
\toprule
Child & Parent & $\pi_k$ & $s_k$ \\
\midrule
%%TCHILDREN%%
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Best solution found at each width in the detailed example (groups of children covered by each latent; a child listed in two groups is coded by both latents), its exact loss, and the prediction from applying the best-division rule repeatedly from width 2.}
\label{tab:split-sweep}
\resizebox{\textwidth}{!}{%
\begin{tabular}{clcl}
\toprule
Width & Best solution found & Loss & Greedy prediction \\
\midrule
%%TSWEEP%%
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[t]
\centering
\caption{Predicting which parent receives the extra latent at width 3, over %%N%% random instances.}
\label{tab:split-predictors}
\small
\begin{tabular}{lc}
\toprule
Predictor (computed from the width-2 solution) & Correct \\
\midrule
Best-division gain, small-$\lambda$ formula & %%DIV0%% / %%N%% \\
Best-division gain, $\lambda=0.05$ & %%DIVLAM%% / %%N%% \\
Frequency-weighted spread $\sum_k\pi_kd_k^2$ & %%SPREAD%% / %%N%% \\
Total frequency of children & %%FREQ%% / %%N%% \\
Number of children & %%NKIDS%% / %%NKIDSN%% \\
\bottomrule
\end{tabular}
\end{table}
"""
pct = lambda v: f"{100 * v:.0f}\\%"
rep = {
    "%%N%%": str(n), "%%DIV0%%": str(acc["div0"]), "%%DIVLAM%%": str(acc["divlam"]),
    "%%SPREAD%%": str(acc["spread"]), "%%SPREADMISS%%": str(n - acc["spread"]),
    "%%FREQ%%": str(acc["freq"]), "%%NKIDS%%": str(acc["nkids"]), "%%NKIDSN%%": str(acc["nkids_n"]),
    "%%MISSRATIO%%": f"{spread_miss_ratio:.2f}",
    "%%FRACMEAN%%": pct(np.mean(frac_best)), "%%FRACMIN%%": pct(min(frac_best)),
    "%%NALT%%": str(len(alt)), "%%ALTMED%%": pct(sorted(alt)[len(alt) // 2]),
    "%%NRUNS%%": str(n_runs), "%%NOVIOL%%": str(n_runs - viol_runs), "%%NVIOL%%": str(viol_runs),
    "%%VMIN%%": f"{100 * min(viol_gaps):.1f}\\%", "%%VMAX%%": f"{100 * max(viol_gaps):.0f}\\%",
    "%%TCHILDREN%%": t_children, "%%TSWEEP%%": t_sweep,
}
tex = TEX
for k, v in rep.items():
    tex = tex.replace(k, v)
assert "%%" not in tex.replace("% ----", ""), "unfilled placeholder"
open(os.path.join(OUT, "appendix_split_order.tex"), "w").write(tex)
print("accuracy:", acc, "| frac mean %.2f min %.2f | alt %d median %.3f | viol %d/%d range %.3f-%.3f | miss ratio %.2f"
      % (np.mean(frac_best), min(frac_best), len(alt), sorted(alt)[len(alt) // 2], viol_runs, n_runs,
         min(viol_gaps), max(viol_gaps), spread_miss_ratio))
for m in range(3, 8):
    print(m, "pred", fmt([[names[k] for k in g] for g in chain[m]]))
