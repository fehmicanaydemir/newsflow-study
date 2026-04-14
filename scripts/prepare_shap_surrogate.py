# scripts/prepare_shap_surrogate.py
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

# import src/*
sys.path.append(str(Path(__file__).parent.parent))
from src.nu_data import load_nu_dataset
from src.data_processing import load_and_preprocess_data
from src.utils import load_recommender, get_user_recommended_item, cosine_scores_from_user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NU", choices=["NU","ML1M","Yahoo","Pinterest"])
    ap.add_argument("--recommender", default="COSINE", choices=["COSINE","MLP","NCF","VAE"])
    ap.add_argument("--emb_csv", type=str, required=True)
    ap.add_argument("--views_csv", type=str, required=True)
    ap.add_argument("--clusters", type=int, default=50, help="number of item clusters")
    ap.add_argument("--eval_users", type=int, default=0, help="limit to first N users (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_name = args.dataset
    rec_name = args.recommender
    files_path = Path("data/processed", data_name)
    files_path.mkdir(parents=True, exist_ok=True)

    # ===== Load data (same path as evaluate.py) =====
    if data_name == "NU":
        (train_data, test_data, static_test_data, pop_array,
         train_array, test_array, items_array, all_items_tensor,
         _pop_dup, item_embs, id2idx, idx2id) = load_nu_dataset(
            args.emb_csv, args.views_csv, device,
            min_profile=5, min_test=5, split_ratio=0.8
        )
        static_test_np = static_test_data.to_numpy() if isinstance(static_test_data, pd.DataFrame) else static_test_data
    else:
        (train_data, test_data, static_test_data, pop_dict,
         train_array, test_array, items_array, all_items_tensor, pop_array) = load_and_preprocess_data(
            data_name, files_path, device
        )
        item_embs = None
        static_test_np = static_test_data if isinstance(static_test_data, np.ndarray) else static_test_data.to_numpy()

    # ===== Limit users if requested =====
    N = test_array.shape[0]
    if args.eval_users and args.eval_users > 0:
        n = min(args.eval_users, N)
        test_array = test_array[:n]
        # only slice test_data if it's a DataFrame
        if isinstance(test_data, pd.DataFrame):
            test_data = test_data.iloc[:n]
        N = n  # update N after trimming

    num_items = items_array.shape[0]

    # ===== Build/normalize item embedding matrix =====
    if item_embs is None:
        raise ValueError("Need item_embs (NU dataset) for cosine SHAP surrogate.")
    item_emb_matrix = np.asarray(item_embs, dtype=np.float32)
    norms = np.linalg.norm(item_emb_matrix, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    item_emb_matrix = item_emb_matrix / norms

    # ===== Cluster items (SHAP surrogate uses clusters) =====
    C = int(args.clusters)
    km = KMeans(n_clusters=C, random_state=0, n_init=10)
    labels = km.fit_predict(item_emb_matrix)  # length = num_items
    item_to_cluster = labels.astype(np.int64)
    np.save(files_path / "item_to_cluster.npy", item_to_cluster)

    # ===== Recommender (COSINE) =====
    kw = {
        "device": device,
        "num_items": num_items,
        "items_array": items_array,
        "all_items_tensor": all_items_tensor,
        "recommender_name": rec_name,
        "item_embs": item_emb_matrix,
        "item_emb_matrix": item_emb_matrix,
    }
    recommender = load_recommender(data_name, hidden_dim=0, recommender_path=None, **kw)

    # ===== Compute per-user cluster "SHAP-like" contributions =====
    shap_values = np.zeros((N, 1 + C), dtype=np.float32)

    item_to_cluster_t = torch.as_tensor(item_to_cluster, device=device)
    item_embs_t = torch.as_tensor(item_emb_matrix, dtype=torch.float32, device=device)

    with torch.no_grad():
        for r in range(N):
            user_vec_np = test_array[r]
            user_tensor = torch.FloatTensor(user_vec_np).to(device)

            # choose uid consistent with metrics: use test_data.index if available; else the row idx
            if isinstance(test_data, pd.DataFrame):
                try:
                    uid = int(test_data.index[r])
                except Exception:
                    uid = int(r)
            else:
                uid = int(r)

            # target item = current top-1 recommendation
            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw).detach().cpu().numpy())

            # zero the target in history (align with your metrics convention)
            user_tensor[item_id] = 0.0

            # baseline score for the target item
            base_scores = cosine_scores_from_user(user_tensor, item_embs_t, None)
            base = float(base_scores[item_id].detach().cpu().numpy())

            # clusters present in user's history
            user_np = user_tensor.detach().cpu().numpy()
            seen_idx = np.flatnonzero(user_np > 0)
            seen_clusters = np.unique(item_to_cluster[seen_idx]) if seen_idx.size > 0 else np.array([], dtype=np.int64)

            contrib = np.zeros(C, dtype=np.float32)
            if seen_clusters.size > 0:
                for c in seen_clusters:
                    mask_user = user_tensor.clone()
                    # zero out all items belonging to cluster c
                    mask = (item_to_cluster_t == int(c))
                    mask_user[mask] = 0.0
                    s = cosine_scores_from_user(mask_user, item_embs_t, None)
                    new = float(s[item_id].detach().cpu().numpy())
                    contrib[int(c)] = max(0.0, base - new)  # positive support from cluster c

            shap_values[r, 0] = uid
            shap_values[r, 1:] = contrib

    np.save(files_path / "shap_values.npy", shap_values)
    print(f"[OK] Saved: {files_path / 'item_to_cluster.npy'} and {files_path / 'shap_values.npy'}")


if __name__ == "__main__":
    main()
