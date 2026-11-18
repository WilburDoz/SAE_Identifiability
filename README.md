These files produce the data plots in the paper "How Optimality Structures Sparse Dictionaries: Theory for Interpreting SAE Representations" (2026). Figure and appendix numbers refer to the revised submission.

Install the dependencies with `pip install -r requirements.txt`.

## Which file makes each figure

| Figure | File(s) |
|---|---|
| Fig. 1A–D | `2_variable_toy_check.ipynb` optimises the dictionaries; `Clean_Plot_Maker.ipynb` generates the data and makes the plots |
| Fig. 2 | `paper_panels.py` → `figures/fig_real_sae_feature_condition.{pdf,png}`, from the real-SAE pipeline below |
| Fig. 3D | `paper_panels.py` → `figures/fig2D_residual_condition_across_datasets.{pdf,png}` |
| Fig. 4A, C | `Clean_Plot_Maker.ipynb` |
| Fig. 4B–C, Fig. 6 | `Single_Variable_Prediction.ipynb` |
| Fig. 5D | `2_variable_toy_check.ipynb` |
| Fig. 7 (App. G) | `absorption_toy/make_appendix_figs.py` |
| App. D tables and examples | `Real_SAE_Checks.ipynb` |

## Toy dictionary-learning experiments (Figs. 1, 4, 5, 6)

These notebooks use JAX and optax. `2_variable_toy_check.ipynb` takes the data and optimises a dictionary on it; `Clean_Plot_Maker.ipynb` generates the data and makes the final plots; `Single_Variable_Prediction.ipynb` optimises the encodings of a single scalar variable.

## Real-SAE experiments (Figs. 2 and 3D, App. D)

These test the feature-feature condition, eq. (6), and the feature-residual condition, eq. (8), on the GPT2-small block-8 SAEs from SAE-lens. We test both on the SAEs' training distribution (OpenWebText) and on eight other datasets.

| File | What it does | Used for |
|---|---|---|
| `sae_conditions.py` | The checks themselves. `FeatureData` holds an SAE's representation. `pairwise_check` runs the pairwise test over all feature pairs. `hull_gauge` runs the test for any set of features (a small linear program). `FeatureData.permuted` builds the activity-permutation null. The module docstring gives the maths. | everything |
| `extract_sae_activations.py` | Runs GPT2-small and the SAEs (the main 24,576-latent SAE and the 768–6,144-latent feature-splitting SAEs) over a dataset. Caches the latents to `~/sae_identifiability_cache/<dataset>/`. | everything |
| `dataset_shift.py` | Pairwise test for every dataset and SAE width: violation counts by type, gauge statistics, and whether each violation replicates in both halves of the tokens. | Fig. 2A–B, App. D.2 |
| `extract_layer_shift.py` | Applies the layer-8 SAE to the residual stream at other layers, on the same tokens, saved in the same format as a dataset (`owt_layer<L>`). | Fig. 2A, C; App. D.3 |
| `joint_violations.py` | Searches for sets of features that violate eq. (6) jointly although every pair passes. Shrinks each to a minimal violating set. | Fig. 2B, App. D.4 |
| `residual_condition.py` | Feature-residual condition: for each feature and each principal direction of the residuals, the smallest λ at which the pair passes, compared with the SAE's training λ. | Fig. 3D, App. D.6 |
| `extract_other_layers.py` | Encodes the same tokens with the other layers' own SAEs, for a pooled cross-layer check. | supplementary check |
| `run_subset_sweep.py` | Compares the test against two nulls (shuffled decoder cosines; independently permuted activity). | supplementary check |
| `paper_panels.py` | Makes the paper figures in `figures/`. | Figs. 2 and 3D |
| `Real_SAE_Checks.ipynb` | Loads the cached results. Tables, supporting figures, and the examples of violations discussed in the text. | App. D |
| `run_real_sae.sh` | Runs all of the above in order. | |

To reproduce, run these from the repository root:
```
conda create -n saes python=3.12 && conda activate saes
pip install -r requirements.txt
bash run_real_sae.sh          # a few hours on a laptop, mostly downloads and LPs
jupyter notebook Real_SAE_Checks.ipynb
```

`SAE_Real.ipynb` is the original exploratory version of the pairwise check. It is superseded by the files above.

## Is the un-absorbed dictionary a local minimum? (Fig. 7, App. G.2)

In `absorption_toy/`. Each model is initialised at the ground-truth (un-absorbed) dictionary and trained by gradient descent on the exact expected loss, which is computed by enumerating every activation pattern with its probability, so there is no sampling noise.

| File | What it does |
|---|---|
| `exact_localmin.py` | Data (the toys of Chanin et al. and Tang et al.'s Def. 3.2, each hierarchical or with an independent control of the same frequencies), feature geometries, magnitudes, the three models (L1 dictionary learning, L1 SAE, TopK SAE) and the descent. Takes a JSON list of settings and prints one JSON line of results per setting. |
| `run_toy_localmin.py` | Runs all 104 settings. |
| `results.jsonl` | The saved results of `run_toy_localmin.py`. |
| `make_appendix_figs.py` | Makes Fig. 7 (`fig_toy_localmin.{pdf,png}`) from `results.jsonl`, re-running the two descents in panel A. |

```
cd absorption_toy
OMP_NUM_THREADS=1 python run_toy_localmin.py > results.jsonl
OMP_NUM_THREADS=1 python make_appendix_figs.py
```

## Which parent latents split first? (supplementary, not in the paper)

In `splitting_order/`. Exact-loss L1 dictionary learning on two parents with mutually exclusive children, as the number of latents grows. Tests a best-division rule for which parent receives the next latent.

| File | What it does |
|---|---|
| `split_order.py` | Best of several restarts at each width; `split_sweep.json` is the detailed example. |
| `split_first.py` | 20 random instances: which parent receives the third latent, against several predictors (`split_first.json`). |
| `split_seeds.py` | Every restart, not just the best: losses, groupings, and whether eq. (6) holds (`split_seeds.json`). |
| `make_split_appendix.py` | Makes `fig_split_order.{pdf,png}` and the tables from the saved results. |

```
cd splitting_order
OMP_NUM_THREADS=1 python split_order.py 2,3,4,5,6,7,8 10 split_sweep.json
OMP_NUM_THREADS=1 python split_first.py 20 split_first.json
OMP_NUM_THREADS=1 python split_seeds.py split_seeds.json
python make_split_appendix.py
```
