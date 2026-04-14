# src/config.py
from pathlib import Path

__all__ = [
    "CHECKPOINTS_ROOT",
    "checkpoints_path",
    "recommender_path_dict",
    "hidden_dim_dict",
    "LXR_checkpoint_dict",
]

# ------------------------------------------------------------------------------------
# Single source of truth for checkpoints location (adjust if you move the repo)
# ------------------------------------------------------------------------------------
CHECKPOINTS_ROOT = Path(__file__).resolve().parent.parent / "checkpoints"
checkpoints_path = str(CHECKPOINTS_ROOT)  # some loaders expect a string folder path

# ------------------------------------------------------------------------------------
# Recommender checkpoints
# ------------------------------------------------------------------------------------
# Only used for MLP/NCF/VAE. COSINE does not need a checkpoint.
recommender_path_dict = {
    # ML1M
    ("ML1M", "VAE"): CHECKPOINTS_ROOT / "VAE_ML1M_0.0007_128_10.pt",
    ("ML1M", "MLP"): CHECKPOINTS_ROOT / "MLP1_ML1M_0.0076_256_7.pt",

    # Yahoo
    ("Yahoo", "VAE"): CHECKPOINTS_ROOT / "VAE_Yahoo_0.0001_128_13.pt",
    ("Yahoo", "MLP"): CHECKPOINTS_ROOT / "MLP2_Yahoo_0.0083_128_1.pt",

    # Pinterest
    ("Pinterest", "VAE"): CHECKPOINTS_ROOT / "VAE_Pinterest_0.0002_32_12.pt",
    ("Pinterest", "MLP"): CHECKPOINTS_ROOT / "MLP_Pinterest_0.0062_512_21_0.pt",

    # NU — the MLP you want to evaluate/explain
    ("NU", "MLP"): CHECKPOINTS_ROOT / "mlp_NU.pt",
    # e.g., ("NU","MLP"): CHECKPOINTS_ROOT / "MLP_NU_BEST_trial0.pt",
}

# ------------------------------------------------------------------------------------
# Hidden dimensions / model sizes
# ------------------------------------------------------------------------------------
# For MLP use a single int. For VAE your code might expect a list like [enc, z]
hidden_dim_dict = {
    ("ML1M", "VAE"): [512, 128],
    ("ML1M", "MLP"): 32,

    ("Yahoo", "VAE"): [512, 128],
    ("Yahoo", "MLP"): 32,

    ("Pinterest", "VAE"): [512, 128],
    ("Pinterest", "MLP"): 512,

    # NU — MUST match the MLP checkpoint above
    ("NU", "MLP"): 256,  # change to 64/128/etc. if your checkpoint differs
}

# ------------------------------------------------------------------------------------
# LXR explainer checkpoints
# ------------------------------------------------------------------------------------
# Mapping: (DATASET, RECOMMENDER) -> (checkpoint_filename, hidden_dim)
# The filename is joined with `checkpoints_path` at load time:
#   Path(checkpoints_path, filename)
#
# Add the ("NU", "COSINE") entry if you want LXR shown for the COSINE recommender.
# Make sure the files actually exist in CHECKPOINTS_ROOT.
LXR_checkpoint_dict = {
    ("ML1M", "VAE"): ('LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426.pt', 128),
    ("ML1M", "MLP"): ('LXR_ML1M_MLP_12_39_64_11.59908096547193_0.1414854294885049.pt', 64),

    ("Yahoo", "VAE"): ('LXR_Yahoo_VAE_neg-1.5pos_combined_19_26_128_18.958765029913238_4.92235962483309.pt', 128),
    ("Yahoo", "MLP"): ('LXR_Yahoo_MLP_neg-pos_combined_last_29_37_128_12.40692505393434_0.19367009952856118.pt', 128),

    ("Pinterest", "VAE"): ('LXR_Pinterest_VAE_0_18_64_3.669673618522336_1.7221734058804223.pt', 64),
    ("Pinterest", "MLP"): ('LXR_Pinterest_MLP_0_5_16_10.059416809308486_0.705778173474644.pt', 16),

    # NU
    ("NU", "MLP"):    ('lxr_explainer_NU.pt', 128),
    ("NU", "COSINE"): ("lxr_explainer_NU_COSINE_MIND.pt", 128),
    ("NU", "COSINE_MIND"): ("lxr_explainer_NU_COSINE_MIND.pt", 128),  # add this file to CHECKPOINTS_ROOT to enable LXR on COSINE
}
