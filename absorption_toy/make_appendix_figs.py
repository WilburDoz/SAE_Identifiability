"""Appendix figure, tables and text for the toy 'is pure hierarchy a local minimum?' experiments.

Reads results.jsonl (produced by exact_localmin.py) and re-runs two short descents for panel A.
Writes: fig_toy_localmin.pdf/.png, appendix_toy_localmin.tex
Usage:  OMP_NUM_THREADS=1 python make_appendix_figs.py      (single thread, ~15 s)
"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", HERE)
sys.path.insert(0, HERE)
import torch
import exact_localmin as E

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (hierarchical, independent)
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#cfcdc8"

rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl")) if l.startswith("{")]
drop = lambda r: 100 * (r["loss_gt"] - r["loss_end"]) / r["loss_gt"]
dabs = lambda r: r["end_absorb"] - r["gt_absorb"]
is_h = lambda r: r["data"].endswith("hier")
fam = lambda r: "topk" if r["model"].startswith("topk") else r["model"]


# ---------------------------------------------------------------- panel A: descent trajectories
def trajectory(kind, data, geom="orth", mags="fixed", lam=0.05, k=None, steps=6000, lr=2e-3, every=1):
    A, P, n, par, kids = E.acts(data, mags)
    F = E.geometry(n, par, kids, geom)
    X = A @ F
    with torch.no_grad():
        L_gt = float(E.Model(kind, F, A, lam, k, 0.0, 0).loss(X, P))
    m = E.Model(kind, F, A, lam, k, 1e-3, 0)
    opt = torch.optim.Adam(m.params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / steps)))
    ts, ys = [], []
    for t in range(steps + 1):
        L = m.loss(X, P)
        if t % every == 0:
            ts.append(t); ys.append(100 * (L_gt - float(L)) / L_gt)
        if t == steps:
            break
        opt.zero_grad(); L.backward(); opt.step(); sched.step(); m.project()
    absorb = E.metrics(m, F, A, X, P, par, kids)[0]
    return np.array(ts), np.array(ys), absorb


torch.set_num_threads(1)
tr = {h: trajectory("sae", f"c4_{h}") for h in ["hier", "indep"]}

# ---------------------------------------------------------------- figure
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
fig, axs = plt.subplots(1, 4, figsize=(5.5, 1.95), constrained_layout=True)


def title(ax, letter, text):
    ax.set_title(f"{letter})  {text}", loc="left", fontweight="bold", color=INK, pad=6)


ax = axs[0]
ax.axhline(0, color=RULE, lw=0.8, zorder=0)
for h, col in [("indep", ORANGE), ("hier", BLUE)]:
    t, y, a = tr[h]
    ax.plot(t[1:], y[1:], color=col, lw=1.6, solid_capstyle="round", zorder=3)
    ax.plot(t[-1], y[-1], "o" if h == "hier" else "s", ms=5, color=col, mec="white", mew=0.8, zorder=4,
            clip_on=False)
    lab = f"cos {a:.2f}"
    ax.annotate(lab, (t[-1], y[-1]), xytext=(-2, 5), textcoords="offset points",
                ha="right", va="bottom", color=INK, fontsize=6.5)
ax.set_xscale("log"); ax.set_xlim(1, 6000); ax.set_ylim(-1, 9.5)
ax.set_xticks([1, 100]); ax.set_xticklabels(["1", "100"]); ax.set_yticks([0, 4, 8])
ax.set_xlabel("descent step")
ax.set_ylabel("loss decrease (%)")
title(ax, "A", "L1 SAE, descent")


def scatter(ax, sel, letter, text, ylim, label_clusters=False, ylabel=True):
    ax.axhline(0, color=RULE, lw=0.8, zorder=0); ax.axvline(0, color=RULE, lw=0.8, zorder=0)
    for h, col, mk in [(False, ORANGE, "s"), (True, BLUE, "o")]:
        S = [r for r in sel if is_h(r) == h]
        ax.scatter([dabs(r) for r in S], [drop(r) for r in S], s=22, c=col, marker=mk,
                   edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xlim(-0.4, 1.0); ax.set_ylim(*ylim)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", ".5", "1"])
    ax.set_xlabel("Δ child–parent cos")
    if ylabel:
        ax.set_ylabel("loss decrease (%)")
    title(ax, letter, text)
    if label_clusters:
        ax.annotate("all 12\ncontrols", (0.0, 0.2), xytext=(-0.37, 8),
                    textcoords="data", fontsize=6.5, color=INK, linespacing=1.1,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=3))


core = [r for r in rows if r["geom"] != "superpos"]
scatter(axs[1], [r for r in core if fam(r) == "dl"], "B", "L1 DL", (-1.5, 31), True)
scatter(axs[2], [r for r in core if fam(r) == "sae"], "C", "L1 SAE", (-1.5, 31), True, ylabel=False)
scatter(axs[3], [r for r in core if fam(r) == "topk"], "D", "TopK SAE", (-4, 92), ylabel=False)

handles = [Line2D([], [], color=BLUE, marker="o", lw=1.6, ms=5, mec="white", label="hierarchical (child only with parent)"),
           Line2D([], [], color=ORANGE, marker="s", lw=1.6, ms=5, mec="white", label="independent control (same frequencies)")]
fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=7.5)
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(OUT, f"fig_toy_localmin.{ext}"), dpi=300)
print("figure written")


# ---------------------------------------------------------------- tables
MODEL = {"dl": "L1 DL", "sae": "L1 SAE", "topk-k1": "TopK ($k{=}1$)", "topk-k2": "TopK ($k{=}2$)"}
DATA = {"c4": "Chanin (4 feat.)", "tang": "Def.~3.2", "c12": "Chanin (12 feat.)"}


def get(model, data, geom="orth", mags="fixed"):
    return next(r for r in rows if r["model"] == model and r["data"] == data and r["geom"] == geom and r["mags"] == mags)


t1 = []
order = [("dl", "c4"), ("dl", "tang"), ("sae", "c4"), ("sae", "tang"),
         ("topk-k1", "c4"), ("topk-k1", "tang"), ("topk-k2", "c12")]
prev = None
for model, d in order:
    if prev is not None and model != prev:
        t1.append("\\midrule")
    prev = model
    for h in ["indep", "hier"]:
        r = get(model, f"{d}_{h}")
        t1.append(f"{MODEL[model]} & {DATA[d]} & {'hier.' if h == 'hier' else 'indep.'} & "
                  f"{r['loss_gt']:.4f} $\\to$ {r['loss_end']:.4f} & {drop(r):.1f}\\% & "
                  f"${r['gt_absorb']:.2f} \\to {r['end_absorb']:.2f}$ & {100 * r['end_par_w_child']:.0f}\\% \\\\")
table1 = r"""\begin{table}[t]
\centering
\caption{Descent from the ground-truth (un-absorbed) dictionary, orthogonal features with unit magnitudes. Loss is the exact expected loss. ``Child--parent cos'' is the cosine between the child latent's decoder and the parent direction (averaged over children for Def.~3.2); ``parent fires'' is the fraction of inputs containing the parent and a child on which the parent latent is active.}
\label{tab:toy-localmin}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllcrcr}
\toprule
Model & Data & Features & Loss: GT $\to$ after & Decrease & Child--parent cos & Parent fires with child \\
\midrule
""" + "\n".join(t1) + r"""
\bottomrule
\end{tabular}}
\end{table}"""


def rng(vals, fmt="{:.1f}"):
    lo, hi = (0.0 if abs(v) < 5e-3 else v for v in (min(vals), max(vals)))
    a, b = fmt.format(lo), fmt.format(hi)
    return f"${a}$" if a == b else f"${a}$ to ${b}$"


t2 = []
for gname, gsel in [("Orthogonal and non-orthogonal ($\\pm0.3$), fixed and variable magnitudes", lambda r: r["geom"] != "superpos"),
                    ("Superposition ($n$ features in $n-1$ dimensions), fixed and variable magnitudes", lambda r: r["geom"] == "superpos")]:
    t2.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{gname}}}}} \\\\")
    for f, fname in [("dl", "L1 DL"), ("sae", "L1 SAE"), ("topk", "TopK")]:
        for h in [True, False]:
            S = [r for r in rows if fam(r) == f and gsel(r) and is_h(r) == h]
            t2.append(f"{fname} & {'hier.' if h else 'indep.'} & {len(S)} & {rng([drop(r) for r in S])}\\% & "
                      f"{rng([dabs(r) for r in S], '{:+.2f}')} & {rng([100 * r['end_par_w_child'] for r in S], '{:.0f}')}\\% \\\\")
    t2.append("\\midrule")
t2 = t2[:-1]
table2 = r"""\begin{table}[t]
\centering
\caption{Robustness across feature geometry and magnitude distribution. Ranges are over settings (geometry $\times$ magnitudes $\times$ dataset). TopK uses $k{=}1$ for the 4- and 6-feature toys and $k{=}2$ for the 12-feature toy.}
\label{tab:toy-localmin-robust}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrccc}
\toprule
Model & Features & Settings & Loss decrease & Change in child--parent cos & Parent fires with child \\
\midrule
""" + "\n".join(t2) + r"""
\bottomrule
\end{tabular}}
\end{table}"""

# ---------------------------------------------------------------- TopK extras: seed sweep and absorption path
def topk_seed(seed, steps=6000, lr=2e-3):
    A, P, n, par, kids = E.acts("c4_hier", "fixed")
    F = E.geometry(n, par, kids, "orth"); X = A @ F
    m = E.Model("topk", F, A, 0.05, 1, 1e-3, seed)
    opt = torch.optim.Adam(m.params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * t / steps)))
    for t in range(steps):
        L = m.loss(X, P); opt.zero_grad(); L.backward(); opt.step(); sched.step(); m.project()
    with torch.no_grad():
        Z = m.codes(X); C = F @ m.Wd
        return dict(loss=float(m.loss(X, P)), child_cos_parent=float(C[0, 1]),
                    parent_cos_child=float(C[1, 0]), child_fires=float((P * (Z[:, 1] > 1e-6)).sum()))


sweep = [topk_seed(sd) for sd in range(6)]
absorbed_runs = [r for r in sweep if r["child_fires"] > 1e-3]
dead_runs = [r for r in sweep if r["child_fires"] <= 1e-3]
L_gt_topk = get("topk-k1", "c4_hier")["loss_gt"]
var_topk_h = [dabs(r) for r in rows if fam(r) == "topk" and r["mags"] == "var" and is_h(r)]
print("TopK seed sweep:", [(round(r["loss"], 4), round(r["child_cos_parent"], 2), round(r["child_fires"], 3)) for r in sweep])

A_, P_, _, _, _ = E.acts("c4_hier", "fixed")
A_, P_ = A_.numpy(), P_.numpy()


def path_loss(dl, k=None, lam=None):
    We = np.eye(4); We[0] = [1, -dl, 0, 0]; We[1] = [0, np.sqrt(1 + dl * dl), 0, 0]
    Wd = np.eye(4); Wd[:, 1] = np.array([dl, 1, 0, 0]) / np.sqrt(1 + dl * dl)
    Z = np.maximum(A_ @ We.T - (lam or 0.0), 0)
    if k is not None:
        drop_idx = np.argsort(-Z, axis=1, kind="stable")[:, k:]
        np.put_along_axis(Z, drop_idx, 0.0, axis=1)
    per = ((A_ - Z @ Wd.T) ** 2).sum(1) + (2 * lam * Z.sum(1) if lam else 0.0)
    return float((P_ * per).sum())


deltas = [0, 0.25, 0.5, 0.75, 1.0]
t3 = []
for dl in deltas:
    cells = [f"{path_loss(dl, k):.4f}" for k in [1, 2, 3, 4]] + [f"{path_loss(dl, lam=0.05):.4f}"]
    t3.append(f"{dl:.2f} & " + " & ".join(cells) + r" \\")
table3 = r"""\begin{table}[t]
\centering
\caption{Exact expected loss along the absorption path of \citet{chanin2025absorption} on the hierarchical four-feature toy: parent encoder $f_0 - \delta f_1$, child encoder $\sqrt{1+\delta^2}\,f_1$, child decoder $(f_1 + \delta f_0)/\sqrt{1+\delta^2}$, other latents fixed at the ground truth. $\delta = 0$ is the un-absorbed and $\delta = 1$ the fully absorbed dictionary. TopK SAEs use no bias and MSE only; the L1 SAE uses encoder bias $-\lambda$ and penalty $2\lambda\|z\|_1$ with $\lambda = 0.05$. When $k$ never binds ($k = 4$) the path is flat at zero loss; whenever $k$ binds, or with an L1 penalty, the loss decreases towards absorption.}
\label{tab:toy-topk-path}
\small
\begin{tabular}{cccccc}
\toprule
$\delta$ & TopK $k{=}1$ & TopK $k{=}2$ & TopK $k{=}3$ & TopK $k{=}4$ & L1 SAE \\
\midrule
""" + "\n".join(t3) + r"""
\bottomrule
\end{tabular}
\end{table}"""

# ---------------------------------------------------------------- appendix text
core_l1 = lambda f, h: [r for r in rows if fam(r) == f and r["geom"] != "superpos" and is_h(r) == h]
H = core_l1("dl", True) + core_l1("sae", True)
I = core_l1("dl", False) + core_l1("sae", False)
sp_dl_h = [r for r in rows if fam(r) == "dl" and r["geom"] == "superpos" and is_h(r)]
sp_dl_i = [r for r in rows if fam(r) == "dl" and r["geom"] == "superpos" and not is_h(r)]
c4h = get("sae", "c4_hier"); tgh = get("sae", "tang_hier"); tgh_dl = get("dl", "tang_hier")
c12h = get("topk-k2", "c12_hier"); c12i = get("topk-k2", "c12_indep")

text = rf"""% ---- generated by make_appendix_figs.py; edit citation keys / labels to match the paper ----
\section{{Is the un-absorbed dictionary a local minimum? Toy experiments}}
\label{{app:toy-localmin}}

Tang et al.\ attribute absorption to spurious minima: the un-absorbed (``pure hierarchical'') dictionary is a good solution, and absorbed patterns are separate minima in which optimisation becomes trapped. This predicts that optimisation started at the un-absorbed dictionary stays there. Eq.~(6) predicts the opposite for L1 dictionary learning: a dictionary in which the child's support is nested inside a distinct parent latent's support is never a local optimum. We test this directly in three models.

\paragraph{{Data.}} (i) The toy of \citet{{chanin2025absorption}}: four features, $f_0$ active with probability $0.25$, $f_1$ with probability $0.05$ but only when $f_0$ is active, and $f_2,f_3$ independently with probability $0.05$. (ii) Definition~3.2 of \citet{{tang2025unified}}: three children active independently with probability $0.05$, a parent active if and only if at least one child is, and two further independent features with probability $0.05$. (iii) The 12-feature TopK toy of \citet{{chanin2025absorption}}: $f_0$ with probability $0.4$, $f_1$ with probability $0.6$ only when $f_0$ is active, and the remaining features with probability $0.15$. Each hierarchical dataset has an independent control with the same firing frequencies. Active magnitudes are either $1$ or uniform on $\{{0.5, 1, 1.5\}}$. Feature directions are orthonormal; non-orthogonal (identity plus $\mathcal N(0, 0.2^2)$ noise, renormalised, with every parent--child cosine set to $+0.3$ or $-0.3$); or in superposition ($n$ random unit vectors in $n-1$ dimensions).

\paragraph{{Models and procedure.}} L1 dictionary learning ($\|x - Wz\|^2 + 2\lambda\|z\|_1$, $z\ge0$, codes optimised per input); a ReLU SAE with the same L1 penalty (encoder bias, no decoder bias); and a TopK SAE trained on reconstruction error alone ($k{{=}}1$ for toys (i)--(ii), $k{{=}}2$ for (iii)). We use $\lambda = 0.05$ and unit-norm decoder columns, imposed by normalising inside the loss. Because each dataset has finitely many activation patterns, we compute the \emph{{exact}} expected loss by enumerating them with their probabilities, so there is no sampling noise. Each model is initialised at the ground truth (decoder equal to the true directions; encoder equal to the dual basis with bias $-\lambda$ for L1 models and $0$ for TopK; codes equal to the soft-thresholded true activations), perturbed by $10^{{-3}}$ Gaussian noise, and optimised with Adam (learning rate $2\times10^{{-3}}$, cosine decay, 6000 steps).

\paragraph{{Results.}} For L1 dictionary learning and the L1 SAE, descent always leaves the ground truth for absorption (Fig.~\ref{{fig:toy-localmin}}, Tables~\ref{{tab:toy-localmin}} and~\ref{{tab:toy-localmin-robust}}). On the Chanin toy the child latent's decoder takes on the parent direction (cosine {c4h['end_absorb']:.2f}), the parent latent stops firing whenever the child is present, and the loss falls by {drop(c4h):.1f}\%. Under Definition~3.2 absorption is partial (cosine {min(tgh['end_absorb'], tgh_dl['end_absorb']):.2f}--{max(tgh['end_absorb'], tgh_dl['end_absorb']):.2f}) and the loss falls by {drop(tgh):.0f}\%. Across all orthogonal and non-orthogonal settings with fixed or variable magnitudes, the child decoder's cosine with the parent rises by {min(map(dabs, H)):.2f}--{max(map(dabs, H)):.2f} and the loss falls by {min(map(drop, H)):.1f}--{max(map(drop, H)):.1f}\%. Variable magnitudes make absorption partial, as \citet{{chanin2025absorption}} found: the parent latent still fires on some child inputs. In every independent control the ground-truth structure is retained (cosine change at most {max(abs(dabs(r)) for r in I):.2f}, parent latent always active, loss change at most {max(map(drop, I)):.1f}\%). Under superposition, L1 dictionary learning shows the same separation in the decoder (cosine increase {min(map(dabs, sp_dl_h)):.2f}--{max(map(dabs, sp_dl_h)):.2f} versus at most {max(abs(dabs(r)) for r in sp_dl_i):.3f} for controls). For the SAE under superposition the dual-basis encoder is far from optimal, so the loss falls substantially in both conditions, but absorption appears only with hierarchy.

For TopK SAEs the ground truth is not a local minimum whenever $k$ binds, for independent as well as hierarchical data. On an input with more than $k$ active features, at least one feature is lost; tilting a selected latent's decoder towards it gives a first-order gain on that input at only a second-order cost on the latent's single-feature inputs. Hierarchy makes such co-occurrence systematic, and because the child is never active alone, nothing limits the tilt, so the child latent can absorb the parent fully. On the 12-feature toy the child decoder ends with cosine {c12h['end_absorb']:.2f} to the parent, against ${c12i['end_absorb']:.2f}$ for the independent control. Descent does not always reach absorption, however. With unit magnitudes and $k{{=}}1$, the parent and child latents tie on parent-plus-child inputs at the ground truth, and only one of them can fire. If the parent latent wins, the child latent, whose feature never appears alone, receives no gradient and dies, and the parent latent can only tilt part of the way towards the child. Across {len(sweep)} seeds on the Chanin toy, {len(absorbed_runs)} reach absorption (loss {min(r['loss'] for r in absorbed_runs):.3f}) and {len(dead_runs)} end with a dead child latent (loss {min(r['loss'] for r in dead_runs):.3f}--{max(r['loss'] for r in dead_runs):.3f}); both outcomes are below the ground truth ({L_gt_topk:.3f}). This is a genuine optimisation trap created by hard top-$k$ selection, but the trapped state is not absorption, and absorption is the better solution. With variable magnitudes the tie is broken differently on different inputs, and every hierarchical TopK run moved into absorption (child--parent cosine increase at least {min(var_topk_h):.2f}). Finally, when $k$ never binds, here meaning $k$ is at least the largest number of co-active features (four, against a mean of $0.4$), the ground truth and the absorbed dictionary both reach zero loss and are joined by a path of zero-loss dictionaries (Table~\ref{{tab:toy-topk-path}}). The objective then prefers neither, so absorption is not favoured, but nor is it a worse minimum. In no model is absorption a spurious minimum that is worse than the un-absorbed solution, and for L1 dictionary learning and the L1 SAE, optimisation started at the ideal dictionary always moves into it.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{fig_toy_localmin.pdf}}
\caption{{The un-absorbed dictionary is not a local minimum. \textbf{{A)}} Exact loss during descent from the ground truth for an L1 SAE on the Chanin toy: with hierarchy the loss keeps falling as the child latent absorbs the parent direction; with independent features it stays at the ground truth (labels give the child decoder's cosine with the parent at the end). \textbf{{B--D)}} End points of descent from the ground truth for every non-superposition setting (orthogonal or $\pm0.3$ parent--child cosine; fixed or variable magnitudes; Chanin and Definition~3.2 toys), showing the change in the child decoder's cosine with the parent against the decrease in loss. For L1 dictionary learning (B) and the L1 SAE (C) every hierarchical setting moves into absorption with lower loss, while every independent control stays at the origin. For TopK (D, $k{{=}}1$ or $2$) descent leaves the ground truth whenever $k$ binds, for both kinds of data; the hierarchical TopK points near zero cosine change are fixed-magnitude runs that did not reach absorption (see text).}}
\label{{fig:toy-localmin}}
\end{{figure}}

{table1}

{table2}

{table3}
"""
open(os.path.join(OUT, "appendix_toy_localmin.tex"), "w").write(text)
print("tex written")
