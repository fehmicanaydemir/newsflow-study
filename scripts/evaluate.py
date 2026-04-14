# scripts/evaluate.py
import argparse
from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd
import torch

# make `src/` importable
sys.path.append(str(Path(__file__).parent.parent))

from src.metrics import eval_one_expl_type, create_results_table
from src.utils import load_recommender
from src.visualization import plot_all_metrics, plot_continuous_metric_distributions
from src.config import recommender_path_dict, hidden_dim_dict

# NU-specific loader
from src.nu_data import load_nu_dataset
# Original repo loader (for ML1M/Yahoo/Pinterest)
from src.data_processing import load_and_preprocess_data


def run_all_baselines(
    data_name,
    recommender_name,
    test_array,
    test_data,
    items_array,
    recommender,
    kw_dict,
    files_path,
    metric_type,
    steps,
    user_indices=None,       # same cohort for all methods
    mask_by="history",       # "history" (discrete) or "all" (continuous)
):
    """
    Run explainer baselines. Automatically skips ones whose prerequisites
    are missing and warns you about what will run.
    The same `user_indices` cohort is used for every method.
    """
    baselines = ["jaccard", "cosine", "lime", "accent", "lxr", "shap"]

    # Cosine requires item embeddings
    if kw_dict.get("item_emb_matrix", None) is None:
        if "cosine" in baselines:
            print("[WARN] No item_emb_matrix -> skipping COSINE baseline.")
            baselines.remove("cosine")

    # LIME/ACCENT/LXR/SHAP require the prebuilt explanation dict file
    expl_dict_path = files_path / f"{recommender_name}_explanation_dicts.pkl"
    if not expl_dict_path.exists():
        need = {"lime", "accent", "lxr", "shap"}
        missing = need.intersection(baselines)
        if missing:
            print(f"[WARN] {expl_dict_path} not found -> skipping {sorted(missing)}.")
            baselines = [b for b in baselines if b not in need]
    else:
        try:
            with open(expl_dict_path, "rb") as f:
                d = pickle.load(f)
            present = set(k.lower() for k in d.keys())
            need = {"lime", "accent", "lxr", "shap"}
            for m in list(need):
                if m in baselines and m not in present:
                    print(f"[WARN] Method '{m}' not present in {expl_dict_path} -> skipping it.")
                    baselines.remove(m)
        except Exception as e:
            print(f"[WARN] Could not read {expl_dict_path} ({e}) -> skipping lime/accent/lxr/shap.")
            baselines = [b for b in baselines if b not in {"lime", "accent", "lxr", "shap"}]

    print(f"[INFO] Baselines to run ({metric_type}, mask_by={mask_by}): {baselines}")

    results = {}
    for baseline in baselines:
        print(f"Running {baseline} baseline for {data_name} {recommender_name} ({metric_type}, mask_by={mask_by})")
        results[baseline] = eval_one_expl_type(
            expl_name=baseline,
            data_name=data_name,
            recommender_name=recommender_name,
            test_array=test_array,
            test_data=test_data,
            items_array=items_array,
            recommender=recommender,
            kw_dict=kw_dict,
            files_path=files_path,
            metric_type=metric_type,
            steps=steps,
            mask_by=mask_by,             # pass masking policy
            user_indices=user_indices,   # same cohort for all methods
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ML1M",
                        help="Dataset to use (ML1M, Yahoo, Pinterest, NU)")
    parser.add_argument("--recommender", type=str, default="MLP",
                        help="Recommender to evaluate (MLP, VAE, NCF, COSINE)")

    # NU-only args (ignored for other datasets)
    parser.add_argument("--emb_csv", type=str, default=None, help="Path to embeddings CSV (NU only)")
    parser.add_argument("--views_csv", type=str, default=None, help="Path to views CSV (NU only)")
    parser.add_argument("--min_profile", type=int, default=5, help="Min train interactions per user (NU)")
    parser.add_argument("--min_test", type=int, default=5, help="Min test interactions per user (NU)")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train split ratio by timestamp (NU)")

    # Evaluate a subset of users
    parser.add_argument("--eval_users", type=int, default=0, help="Evaluate on N users (0 = all)")
    parser.add_argument("--random_eval", action="store_true",
                        help="When --eval_users>0, pick a random subset instead of first N")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for --random_eval")

    args = parser.parse_args()

    data_name = args.dataset
    recommender_name = args.recommender

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files_path = Path("data/processed", data_name)

    # ===== Load data =====
    if data_name == "NU":
        (train_data, test_data, static_test_data, pop_array,
         train_array, test_array, items_array, all_items_tensor,
         _pop_array_dup, item_embs, id2idx, idx2id) = load_nu_dataset(
            args.emb_csv, args.views_csv, device,
            min_profile=args.min_profile, min_test=args.min_test, split_ratio=args.split_ratio
        )
        static_test_np = static_test_data.to_numpy() if isinstance(static_test_data, pd.DataFrame) else static_test_data
    else:
        (train_data, test_data, static_test_data, pop_dict,
         train_array, test_array, items_array, all_items_tensor, pop_array) = load_and_preprocess_data(
            data_name, files_path, device
        )
        item_embs = None
        static_test_np = static_test_data if isinstance(static_test_data, np.ndarray) else static_test_data.to_numpy()

    # ===== Optional subsample (keep mapping row->original user) =====
    orig_idx = np.arange(test_array.shape[0], dtype=np.int64)
    if args.eval_users and args.eval_users > 0:
        n_total = test_array.shape[0]
        n = min(args.eval_users, n_total)
        if args.random_eval:
            rng = np.random.default_rng(args.seed)
            chosen = rng.choice(n_total, size=n, replace=False)
        else:
            chosen = np.arange(n, dtype=np.int64)
        chosen = np.sort(chosen)

        test_array = test_array[chosen]
        orig_idx   = orig_idx[chosen]
        if isinstance(static_test_data, pd.DataFrame):
            static_test_data = static_test_data.iloc[chosen]
            static_test_np   = static_test_data.to_numpy()
        elif isinstance(static_test_np, np.ndarray):
            static_test_np   = static_test_np[chosen]
        if isinstance(test_data, pd.DataFrame):
            test_data = test_data.iloc[chosen]

    num_items = items_array.shape[0]

    # ===== Item embeddings for cosine =====
    if data_name == "NU" and item_embs is not None:
        item_emb_matrix = np.asarray(item_embs, dtype=np.float32)
        norms = np.linalg.norm(item_emb_matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8
        item_emb_matrix = item_emb_matrix / norms
    else:
        item_emb_matrix = None

    # ===== Jaccard dict (optional) =====
    jaccard_path = files_path / f"jaccard_based_sim_{data_name}.pkl"
    if jaccard_path.exists():
        with open(jaccard_path, "rb") as f:
            jaccard_dict = pickle.load(f)
    else:
        print(f"[WARN] No Jaccard file at {jaccard_path}; Jaccard baseline will be zero.")
        jaccard_dict = {}

    # ===== Shared kwargs =====
    kw_dict = {
        "device": device,
        "num_items": num_items,
        "num_features": num_items,
        "demographic": False,
        "pop_array": pop_array,
        "all_items_tensor": all_items_tensor,
        "static_test_data": static_test_data if data_name != "NU" else static_test_np,
        "items_array": items_array,
        "recommender_name": recommender_name,
        # LIME config
        "min_pert": 80,
        "max_pert": 200,
        "num_of_perturbations": 600,
        # Similarity dicts / matrices
        "jaccard_dict": jaccard_dict,
        "item_embs": item_emb_matrix,        # COSINE wrapper
        "item_emb_matrix": item_emb_matrix,  # cosine explainer
        "k_masked": 1,
        # mapping (local row -> original user id) for dict-based explainers
        "user_index_map": orig_idx,
    }

    # Paper-style: evaluate ranks over ALL items so endpoints match at 100%
    kw_dict["exclude_seen_eval"] = False

    # ===== Build recommender (must be before running baselines) =====
    if recommender_name == "COSINE":
        recommender_path = None
        hidden_dim = 0
    else:
        recommender_path = recommender_path_dict[(data_name, recommender_name)]
        hidden_dim = hidden_dim_dict[(data_name, recommender_name)]
    recommender = load_recommender(data_name, hidden_dim, recommender_path, **kw_dict)

    # ===== Use ONE shared cohort across all methods =====
    # You already subsampled/ordered above; just evaluate over all rows now.
    eval_rows = np.arange(test_array.shape[0], dtype=np.int64)

    # ===== DISCRETE (Ke=1..5) over HISTORY items =====
    discrete_results = run_all_baselines(
        data_name=data_name,
        recommender_name=recommender_name,
        test_array=test_array,
        test_data=test_data,
        items_array=items_array,
        recommender=recommender,
        kw_dict=kw_dict,
        files_path=files_path,
        metric_type="discrete",
        steps=5,
        user_indices=eval_rows,    # same cohort
        mask_by="history",         # history-only masking (paper)
    )
    if discrete_results:
        df_discrete = create_results_table(discrete_results, data_name, recommender_name)
        print("Discrete Metrics Results:")
        print(df_discrete)
        plot_all_metrics(discrete_results, data_name, recommender_name, metric_type="discrete", steps=5)
    else:
        print("[WARN] No baselines ran for discrete mode (check prerequisites).")

    # ===== CONTINUOUS (0..100%) over GLOBAL pool =====
    continuous_results = run_all_baselines(
        data_name=data_name,
        recommender_name=recommender_name,
        test_array=test_array,
        test_data=test_data,
        items_array=items_array,
        recommender=recommender,
        kw_dict=kw_dict,
        files_path=files_path,
        metric_type="continuous",
        steps=11,                  # 0%,10%,...,100%
        user_indices=eval_rows,    # same cohort
        mask_by="all",             # global masking (paper)
    )
    if continuous_results:
        df_continuous = create_results_table(continuous_results, data_name, recommender_name)
        print("\nContinuous Metrics Results:")
        print(df_continuous)
        plot_all_metrics(continuous_results, data_name, recommender_name, metric_type="continuous", steps=11)
        plot_continuous_metric_distributions(continuous_results, data_name, recommender_name)
    else:
        print("[WARN] No baselines ran for continuous mode (check prerequisites).")


if __name__ == "__main__":
    main()
