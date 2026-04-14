# scripts/make_shap_values.py
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import pandas as pd

# make src importable
sys.path.append(str(Path(__file__).parent.parent))
from src.nu_data import load_nu_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_csv", required=True)
    ap.add_argument("--views_csv", required=True)
    ap.add_argument("--item_to_cluster", default="data/processed/NU/item_to_cluster.npy")
    ap.add_argument("--out_dir", default="data/processed/NU")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (to get test users and consistent user_id mapping)
    (train_data, test_data, static_test_data, pop_array,
     train_array, test_array, items_array, all_items_tensor,
     _pop_dup, item_embs, id2idx, idx2id) = load_nu_dataset(
        args.emb_csv, args.views_csv, device,
        min_profile=5, min_test=5, split_ratio=0.8
    )

    # Load item->cluster
    item_to_cluster = np.load(args.item_to_cluster).astype(np.int64)
    K = int(item_to_cluster.max() + 1)
    num_items = items_array.shape[0]
    assert item_to_cluster.shape[0] == num_items, \
        f"item_to_cluster has length {item_to_cluster.shape[0]}, expected {num_items}"

    # We will build shap_values: shape [num_users, 1+K]; first col is user_id
    N = test_array.shape[0]
    shap_values = np.zeros((N, 1 + K), dtype=np.float32)

    for i in range(N):
        # user_id consistent with your build_explanation_dicts.py / metrics.py
        if isinstance(test_data, pd.DataFrame) and hasattr(test_data, "index"):
            user_id = int(test_data.index[i])
        else:
            user_id = int(i)

        uv = test_array[i].astype(np.float32)
        seen_items = np.flatnonzero(uv > 0.0)

        h = np.zeros(K, dtype=np.float32)
        if seen_items.size > 0:
            cl = item_to_cluster[seen_items]
            # histogram per cluster
            for c in cl:
                h[int(c)] += 1.0
            # normalize to sum=1 for interpretability
            s = h.sum()
            if s > 0:
                h = h / s

        shap_values[i, 0] = float(user_id)
        shap_values[i, 1:] = h

    np.save(out_dir / "shap_values.npy", shap_values)
    print(f"[OK] Saved {out_dir / 'shap_values.npy'} ; shape = {shap_values.shape}, K={K}")

if __name__ == "__main__":
    main()
