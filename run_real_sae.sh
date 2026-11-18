#!/usr/bin/env bash
# Regenerates every cached result used by Real_SAE_Checks.ipynb. Results go to ~/sae_identifiability_cache.
# Roughly: extraction ~2 min per dataset (mostly download), dataset_shift ~5 min per SAE width,
# joint_violations ~15 min, run_subset_sweep ~10 min per null (10 cores).
set -e

DATASETS="openwebtext openwebtext_b c4 wiki pile code code_python wiki_de wiki_fr"

# 0. German and French Wikipedia: one parquet shard each (streaming these through `datasets` stalls)
mkdir -p ~/sae_identifiability_cache/raw
curl -sL -o ~/sae_identifiability_cache/raw/wiki_de.parquet \
    https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.de/train-00002-of-00020.parquet
curl -sL -o ~/sae_identifiability_cache/raw/wiki_fr.parquet \
    https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.fr/train-00002-of-00017.parquet

# 1. GPT2-small activations -> SAE latents, for the main SAE and the 768-6144 feature-splitting SAEs
for ds in $DATASETS; do
    python extract_sae_activations.py --dataset "$ds" --shuffle_buffer 500
done

# 2. Pairwise condition on every dataset, for every SAE width ("" = main 24576-latent SAE)
for sae in "" fs_768 fs_1536 fs_3072 fs_6144; do
    python dataset_shift.py --sae "$sae" --datasets $DATASETS
done

# 3. Joint (beyond-pairs) violations, main SAE
python joint_violations.py --datasets $DATASETS

# 4. Feature-residual condition (eq. 8), every SAE width
python residual_condition.py --datasets $DATASETS

# 5. Layer shift: the layer-8 SAE applied to other layers' residual streams (same OWT tokens), plus
#    other layers' own SAEs for the pooled cross-layer check (see Real_SAE_Checks.ipynb)
python extract_layer_shift.py --layers 4 6 7 9 10
python dataset_shift.py --datasets owt_layer4 owt_layer6 owt_layer7 owt_layer9 owt_layer10
python residual_condition.py --saes "" --datasets owt_layer4 owt_layer6 owt_layer7 owt_layer9 owt_layer10
python extract_other_layers.py --layers 4 6 10

# 6. Nulls: what the condition is sensitive to (OpenWebText, main SAE)
python run_subset_sweep.py --dataset openwebtext --null cos
python run_subset_sweep.py --dataset openwebtext --null perm

# 7. Paper figures (feature-condition figure, Fig 2D) -> figures/
python paper_panels.py
