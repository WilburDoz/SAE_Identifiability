"""Makes the real-SAE figures for the paper from the cached results (run run_real_sae.sh first).

    figures/fig_real_sae_feature_condition.{pdf,png}   (end of the feature-feature section)
        eq. (6) on the main SAE. A: violating feature pairs when the layer-8 SAE is applied to the residual
        stream at other layers (same tokens). B: violating pairs and violating feature sets across datasets,
        ordered by the SAE's fraction of variance unexplained. C: one violating pair, on the same tokens at
        layer 8 (passes: the child explores the parent-off line) and layer 6 (fails: it never does).
    figures/fig2D_residual_condition_across_datasets.{pdf,png}   (last panel of the residual figure)
        eq. (8): the residual break ratio lambda* / lambda_train per dataset, one marker per SAE width.
        lambda* is the smallest lambda at which the SAE's residuals satisfy the condition, so > 1 is a violation.

Style matches Clean_Plot_Maker.ipynb.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

CACHE_DIR = os.path.expanduser("~/sae_identifiability_cache")
OUT_DIR = "figures"
LABEL = {"openwebtext": "OWT", "openwebtext_b": "OWT (2)", "c4": "C4", "wiki": "Wiki", "pile": "Pile",
         "wiki_de": "Wiki (de)", "wiki_fr": "Wiki (fr)", "code": "Code", "code_python": "Python"}
WIDTHS = [("fs_768", 768), ("fs_1536", 1536), ("fs_3072", 3072), ("fs_6144", 6144), ("main_24576", 24576)]


def style(ax, labelsize=30):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_linewidth(6)
    ax.tick_params(axis="both", labelsize=labelsize, width=4, length=10)


def dataset_order(shift):
    names = [d for d in LABEL if d in shift]
    return sorted(names, key=lambda d: shift[d]["summary"]["fvu"])


def fig2D(shift, residual):
    fs = 45  # text size (1.5x Clean_Plot_Maker's 30, so it stays legible as one panel of a wide figure)
    order = dataset_order(shift)
    x = np.arange(len(order))
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(WIDTHS)))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axhline(1, color="k", lw=4, ls="--", zorder=-1)
    offsets = np.linspace(-0.24, 0.24, len(WIDTHS))
    for (sae, width), col, dx in zip(WIDTHS, colors, offsets):
        y = [residual[f"{sae}/{d}"]["lam_star"] / residual[f"{sae}/{d}"]["lam_train"]
             if f"{sae}/{d}" in residual else np.nan for d in order]
        ax.plot(x + dx, y, "o", color=col, ms=27, label=f"{width}")
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xticks(x, [LABEL[d] for d in order], rotation=45, ha="right")
    ax.set_ylabel("residual break ratio (> 1 violates)", fontsize=fs)  # = lambda* / lambda_train, see App D
    ax.legend(fontsize=36, title="latents", title_fontsize=36, frameon=False, loc="center left",
              bbox_to_anchor=(1.02, 0.5), handletextpad=0.2)
    style(ax, labelsize=fs)
    return fig


def fig_feature_condition(shift, joint, example=("owt_layer6", 16021, 12423)):
    """Feature-feature condition on real SAEs, four equal panels: (A) the layer-8 SAE applied to other layers'
    residual streams, (B) across datasets, (C, D) one pair on the same tokens at layer 8 (passes) and at
    layer 6 (fails)."""
    import scipy.sparse as sp

    fs, small = 44, 32  # axis/tick text, and annotation text
    box = dict(facecolor="white", edgecolor="none", alpha=0.9)
    fig, axs = plt.subplots(1, 4, figsize=(46, 11), gridspec_kw=dict(wspace=0.4))
    count = lambda d: shift[d]["summary"]["n_violations"]
    n_sets = lambda d: len(joint[d]["violations"]) if d in joint else np.nan

    # A: layer shift. Layer 8 is the SAE's training layer (the "openwebtext" sample itself).
    ax = axs[0]
    layers = [4, 6, 7, 8, 9, 10]
    names = [f"owt_layer{L}" if L != 8 else "openwebtext" for L in layers]
    ax.axvspan(7.6, 8.4, color="0.9", zorder=-2)
    ax.text(8, 280, "trained\nlayer", ha="center", va="top", fontsize=small)
    ax.plot(layers, [count(n) for n in names], "o", ms=32)
    ax.set_xticks(layers)
    ax.set_xlabel("layer the SAE is applied to", fontsize=fs)
    ax.set_ylabel("violating feature pairs", fontsize=fs)

    # B: dataset shift, ordered by FVU
    ax = axs[1]
    order = dataset_order(shift)
    x = np.arange(len(order))
    ax.plot(x - 0.12, [count(d) for d in order], "o", ms=32, label="feature pairs")
    ax.plot(x + 0.12, [n_sets(d) for d in order], "s", ms=29, label="feature sets")
    ax.set_xticks(x, [LABEL[d] for d in order], rotation=45, ha="right")
    ax.set_ylabel("violations", fontsize=fs)
    ax.legend(fontsize=small, frameon=False, loc="upper left")
    for ax in axs[:2]:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(-0.3, 300)
        ax.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
        ax.minorticks_off()

    # C, D: one pair on the same tokens, at the training layer and at a shifted layer, in the paper's
    # coordinates z_bar / |min z_bar|. Parent-off tokens sit on the line x = -1; the red bar is the range
    # (convex hull) the child explores there, which must contain the required point -cos(theta).
    ds, parent, child = example
    W = np.load(f"{CACHE_DIR}/openwebtext/sae_params.npz")["W_dec"]
    cos = W[parent] @ W[child] / np.linalg.norm(W[parent]) / np.linalg.norm(W[child])
    norm = lambda z: (z - z.mean()) / z.mean()  # = z_bar / |min z_bar| when min z = 0
    for ax, name, title in [(axs[2], "openwebtext", "layer 8 (trained)"), (axs[3], ds, f"layer {ds[-1]}")]:
        Z = sp.load_npz(f"{CACHE_DIR}/{name}/feature_acts.npz").tocsc()
        zp, zc = (Z[:, f].toarray().ravel() for f in (parent, child))
        xp, yc = norm(zp), norm(zc)
        ax.plot(xp, yc, "o", ms=11, alpha=0.3, color="C0")
        hull = yc[zp == 0]
        ax.plot([-1, -1], [hull.min(), hull.max()], color="red", lw=14, solid_capstyle="butt", zorder=2)
        ax.plot([-1], [hull.min()], "o", color="red", ms=24, zorder=2)  # visible even when the range collapses
        ax.plot([-1], [-cos], "*", color="k", ms=44, zorder=3)
        ax.annotate(r"required $-\cos\theta$", (-1, -cos), xytext=(1.5, 4), fontsize=small, bbox=box,
                    arrowprops=dict(arrowstyle="-", lw=3))
        ax.set_xlim(-2.5, 12)
        ax.set_ylim(-4, 30)
        ax.set_title(title, fontsize=fs)
        ax.set_xlabel(f"parent feature {parent}", fontsize=fs)
        ax.set_ylabel(f"child feature {child}", fontsize=fs)
    axs[2].text(0.5, 25, "hull (red) covers\nthe required point", fontsize=small, color="red", bbox=box)
    axs[3].text(0.5, 23, "hull collapses:\nchild never fires\nwhile parent is off", fontsize=small,
                color="red", bbox=box)
    for ax in axs:
        style(ax, labelsize=fs)
    axs[1].tick_params(axis="x", labelsize=34)  # nine dataset names need to fit under one panel
    for ax, letter in zip(axs[:3], "ABC"):  # C labels the pair of example panels
        ax.text(-0.28, 1.08, f"{letter})", transform=ax.transAxes, fontsize=64, fontweight="bold", va="bottom")
    return fig


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    shift = json.load(open(f"{CACHE_DIR}/dataset_shift_main.json"))
    joint = json.load(open(f"{CACHE_DIR}/joint_violations_main.json"))
    residual = json.load(open(f"{CACHE_DIR}/residual_condition.json"))
    for name, fig in [("fig_real_sae_feature_condition", fig_feature_condition(shift, joint)),
                      ("fig2D_residual_condition_across_datasets", fig2D(shift, residual))]:
        for ext in ["pdf", "png"]:
            fig.savefig(f"{OUT_DIR}/{name}.{ext}", dpi=100, bbox_inches="tight")
        print("saved", f"{OUT_DIR}/{name}")
