These files produce the data plots from the paper "How Optimality Structures Sparse Dictionaries: A Theory for Understanding SAE Representations" 2026.

## Toy dictionary-learning experiments

Figs 1A - D are produced by a combination of 2_variable_toy_check.ipynb (which takes the data and optimises a dictionary on it) and Clean_Plot_Maker, which generates the data, and later makes the plots that look clean. (Including Fig 3A and C)

The optimisations shown in Fig 3B and C were performed using Single_Variable_Prediction

Finally, Fig 4D was generated using 2_variable_toy_check

## Real SAE experiments (the real-SAE feature-condition figure, Fig 2D, Appendix D)

These test the feature-feature condition, eq. (6), and the feature-residual condition, eq. (8), on the GPT2-small block-8 SAEs from SAE-lens. We test both on the SAEs' training distribution (OpenWebText) and on eight other datasets.

| File | What it does | Used for |
|---|---|---|
| `sae_conditions.py` | The checks themselves. `FeatureData` holds an SAE's representation. `pairwise_check` runs the pairwise test over all feature pairs. `hull_gauge` runs the test for any set of features (a small linear program). `FeatureData.permuted` builds the activity-permutation null. The module docstring gives the maths. | everything |
| `extract_sae_activations.py` | Runs GPT2-small and the SAEs (the main 24,576-latent SAE and the 768–6,144-latent feature-splitting SAEs) over a dataset. Caches the latents to `~/sae_identifiability_cache/<dataset>/`. | everything |
| `dataset_shift.py` | Pairwise test for every dataset and SAE width: violation counts by type, gauge statistics, and whether each violation replicates in both halves of the tokens. | feature figure A–B, App D.2–D.4 |
| `joint_violations.py` | Searches for sets of features that violate eq. (6) jointly although every pair passes. Shrinks each to a minimal violating set. | feature figure B, App D.5 |
| `extract_layer_shift.py` | Applies the layer-8 SAE to the residual stream at other layers, on the same tokens, saved in the same format as a dataset (`owt_layer<L>`). | feature figure A, C; App D.3 |
| `extract_other_layers.py` | Encodes the same tokens with the other layers' own SAEs, for the pooled cross-layer check. | App D.3 |
| `run_subset_sweep.py` | Compares the test against two nulls (shuffled decoder cosines; independently permuted activity). | App D.7 |
| `residual_condition.py` | Feature-residual condition: for each feature and each principal direction of the residuals, the smallest λ at which the pair passes, compared with the SAE's training λ. | Fig 2D, App D.8 |
| `paper_panels.py` | Makes the paper figures in `figures/`. | feature figure, Fig 2D |
| `Real_SAE_Checks.ipynb` | Loads the cached results. Tables, supporting figures, and the examples of violations discussed in the text. | App D |
| `run_real_sae.sh` | Runs all of the above in order. | |

To reproduce, run these from the repository root:
```
conda create -n saes python=3.12 && conda activate saes
pip install torch sae_lens transformer_lens datasets scipy matplotlib tqdm ipykernel
bash run_real_sae.sh          # a few hours on a laptop, mostly downloads and LPs
jupyter notebook Real_SAE_Checks.ipynb
```

`SAE_Real.ipynb` is the original exploratory version of the pairwise check. It is superseded by the files above.
