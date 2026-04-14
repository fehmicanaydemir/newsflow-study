# scripts/prep_processed.py
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import sys

# Make src importable
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.nu_data import load_nu_dataset  # uses your custom split incl. max 100 reads

# Use scipy.sparse for memory-efficient item–item sims
try:
    from scipy.sparse import csr_matrix
except Exception as e:
    raise ImportError("scipy is required for sparse item–item computations. Install with: pip install scipy") from e


def compute_topk_item_item_sims(train_array: np.ndarray, topk: int = 25):
    """
    Compute Top-K item–item cosine and jaccard similarities from a binary user×item matrix.
    Returns two *pruned* dicts with keys (i, j) and values float, storing only topk neighbors for each i.
    """
    # Ensure binary float32 CSR
    X = csr_matrix((train_array > 0).astype(np.float32))
    I = X.shape[1]

    # Item supports: number of users who interacted with each item
    supp = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)  # [I]

    # Co-occurrences: C = X^T X (symmetric, sparse)
    C = (X.T @ X).tocsr()  # shape [I, I]

    cosine_pruned = {}
    jaccard_pruned = {}

    for i in range(I):
        row = C.getrow(i)
        cols = row.indices
        inter = row.data.astype(np.float32)  # intersections with i

        # Skip degenerate items
        if supp[i] <= 0 or cols.size == 0:
            continue

        # Cosine: inter / sqrt(supp[i] * supp[j])
        denom_cos = np.sqrt(supp[i] * supp[cols]) + 1e-8
        cos_vals = inter / denom_cos

        # Jaccard: inter / (supp[i] + supp[j] - inter)
        denom_jac = (supp[i] + supp[cols] - inter) + 1e-8
        jac_vals = inter / denom_jac

        # Exclude self
        mask_not_self = cols != i
        cols = cols[mask_not_self]
        cos_vals = cos_vals[mask_not_self]
        jac_vals = jac_vals[mask_not_self]

        if cols.size == 0:
            continue

        # Select Top-K neighbors by cosine (you can also rank by jaccard—cosine is typical)
        k = min(topk, cols.size)
        top_idx = np.argpartition(-cos_vals, k - 1)[:k]
        nbrs = cols[top_idx]
        cos_top = cos_vals[top_idx]
        jac_top = jac_vals[top_idx]

        # Store directed entries (i -> j). We'll symmetrize on read if needed.
        for j, c, ja in zip(nbrs, cos_top, jac_top):
            cosine_pruned[(int(i), int(j))] = float(c)
            jaccard_pruned[(int(i), int(j))] = float(ja)

    return jaccard_pruned, cosine_pruned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)      # e.g., NU
    ap.add_argument("--emb_csv", required=True)
    ap.add_argument("--views_csv", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=25, help="Top-K neighbors to keep per item")
    # (Optional) if you ever want to drop ultra-rare items, expose a flag here; default keeps all.
    args = ap.parse_args()

    data_name = args.dataset
    device = torch.device(args.device)

    out_dir = Path("data", "processed", data_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load via your nu_data.py (already enforces: drop users <10 or >100 reads; 10–25 → 5 test, >25 → 80/20)
    (train_data, test_data, static_test_data, pop_array,
     train_array, test_array, items_array, all_items_tensor,
     _pop_dup, item_embs, id2idx, idx2id) = load_nu_dataset(
        args.emb_csv, args.views_csv, device, min_profile=5, min_test=5, split_ratio=0.8
    )

    # Save matrices expected by create_dictionaries.py
    pd.DataFrame(train_array).to_csv(out_dir / f"train_data_{data_name}.csv")
    pd.DataFrame(test_array).to_csv(out_dir / f"test_data_{data_name}.csv")
    static_test = pd.DataFrame(
        np.concatenate([test_array, np.zeros((test_array.shape[0], 2), dtype=np.float32)], axis=1)
    )
    static_test.to_csv(out_dir / f"static_test_data_{data_name}.csv", index=True)

    # Popularity dict
    pop_dict = {int(i): float(pop_array[i]) for i in range(len(pop_array))}
    with open(out_dir / f"pop_dict_{data_name}.pkl", "wb") as f:
        pickle.dump(pop_dict, f)

    # Item–item sims (sparse Top-K)
    print(f"Computing sparse Top-{args.topk} item–item similarities...")
    jac_dict, cos_dict = compute_topk_item_item_sims(train_array, topk=args.topk)

    with open(out_dir / f"jaccard_based_sim_{data_name}.pkl", "wb") as f:
        pickle.dump(jac_dict, f)
    with open(out_dir / f"cosine_based_sim_{data_name}.pkl", "wb") as f:
        pickle.dump(cos_dict, f)

    print("Processed files written to:", out_dir)


if __name__ == "__main__":
    main()
