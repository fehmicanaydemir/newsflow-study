# scripts/make_item_clusters.py
import argparse
from pathlib import Path
import sys
import numpy as np
from sklearn.cluster import KMeans
import torch
import pandas as pd

# make src importable
sys.path.append(str(Path(__file__).parent.parent))
from src.nu_data import load_nu_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_csv", required=True)
    ap.add_argument("--views_csv", required=True)
    ap.add_argument("--clusters", type=int, default=50, help="number of item clusters")
    ap.add_argument("--out_dir", type=str, default="data/processed/NU")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load NU dataset just to get item_embs and num_items
    (_, _, _, _,
     _, _, items_array, _,
     _, item_embs, _, _) = load_nu_dataset(
        args.emb_csv, args.views_csv, device,
        min_profile=5, min_test=5, split_ratio=0.8
    )
    item_embs = np.asarray(item_embs, dtype=np.float32)
    print(f"[INFO] item_embs shape = {item_embs.shape}")

    # KMeans clustering
    K = int(args.clusters)
    print(f"[INFO] Clustering items into K={K} clusters...")
    labels = KMeans(n_clusters=K, n_init="auto", random_state=0).fit_predict(item_embs)
    labels = labels.astype(np.int64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "item_to_cluster.npy", labels)
    print(f"[OK] Saved {out_dir / 'item_to_cluster.npy'}")

if __name__ == "__main__":
    main()
