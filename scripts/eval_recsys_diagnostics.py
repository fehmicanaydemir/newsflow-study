# scripts/eval_recsys_diagnostics.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# make src importable
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.nu_data import load_nu_dataset
from src.utils import load_recommender, recommender_run

# ------------------------------
# Metric helpers (implicit data)
# ------------------------------
def precision_at_k(recommended, relevant, k):
    if k == 0: return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant)
    return hits / float(k)

def recall_at_k(recommended, relevant, k):
    if not relevant: return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant)
    return hits / float(len(relevant))

def hit_rate_at_k(recommended, relevant, k):
    if not relevant: return 0.0
    rec_k = recommended[:k]
    return 1.0 if any(i in relevant for i in rec_k) else 0.0

def ndcg_at_k(recommended, relevant, k):
    """Binary relevance DCG with log2 discount, ideal DCG by |relevant ∩ topK|."""
    rec_k = recommended[:k]
    gains = []
    for rank, item in enumerate(rec_k, start=1):
        if item in relevant:
            gains.append(1.0 / np.log2(rank + 1))
    dcg = np.sum(gains)
    # ideal DCG is when all min(m,k) relevant are at the top ranks
    m = min(len(relevant), k)
    if m == 0: 
        return 0.0
    ideal = np.sum([1.0 / np.log2(r + 1) for r in range(1, m + 1)])
    return float(dcg / ideal) if ideal > 0 else 0.0

def apk(recommended, relevant, k):
    """Average Precision @ K (binary relevance)."""
    if not relevant: return 0.0
    score = 0.0; hits = 0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / float(i)
    denom = min(len(relevant), k)
    return score / float(denom) if denom > 0 else 0.0

def mrr_at_k(recommended, relevant, k):
    """Reciprocal rank of first hit in top-K; 0 if none."""
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / float(i)
    return 0.0


# ------------------------------
# Ranking helper
# ------------------------------
@torch.no_grad()
def rank_all_items(user_vec, model, items_array, exclude_seen_mask, device):
    """
    Returns list of item ids sorted by score desc, excluding seen (mask True).
    - user_vec: torch.FloatTensor [I] (train-only profile)
    - model: recommender (COSINE wrapper or MLP)
    - items_array: np.ndarray [I, I] one-hot
    - exclude_seen_mask: np.ndarray[bool] length I (True = exclude)
    """
    # get full score vector
    scores_t = recommender_run(user_vec, model, output_type="vector",
                               items_array=items_array, device=device, num_items=items_array.shape[0])
    scores = scores_t.detach().cpu().numpy()
    # exclude seen
    scores = scores.copy()
    scores[exclude_seen_mask] = -1e9
    order = np.argsort(-scores)
    return order.tolist(), scores[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NU", choices=["NU"])
    ap.add_argument("--recommender", default="COSINE", choices=["COSINE", "MLP"])
    ap.add_argument("--emb_csv", type=str, required=True, help="NU only: embeddings CSV")
    ap.add_argument("--views_csv", type=str, required=True, help="NU only: views CSV")
    ap.add_argument("--eval_users", type=int, default=0, help="limit users (0 = all)")
    ap.add_argument("--ks", type=str, default="5,10,20", help="comma-separated K values")
    ap.add_argument("--outdir", type=str, default="results/diagnostics")
    args = ap.parse_args()

    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- Load NU data (train/test split is inside load_nu_dataset) --------
    (train_data, test_data, static_test_data, pop_array,
     train_array, test_array, items_array, all_items_tensor,
     _pop_dup, item_embs, id2idx, idx2id) = load_nu_dataset(args.emb_csv, args.views_csv, device)

    num_users = train_array.shape[0]
    num_items = items_array.shape[0]

    # Optional subset of users
    idx_users = np.arange(num_users)
    if args.eval_users and args.eval_users > 0:
        idx_users = idx_users[: args.eval_users]

    # -------- Build recommender --------
    if args.recommender == "COSINE":
        model = load_recommender("NU", hidden_dim=0, recommender_path=None,
                                 item_embs=item_embs, recommender_name="COSINE",
                                 device=device, items_array=items_array,
                                 all_items_tensor=all_items_tensor, num_items=num_items)
    else:
        from src.config import recommender_path_dict, hidden_dim_dict
        path = recommender_path_dict[("NU", "MLP")]
        hdim = hidden_dim_dict[("NU", "MLP")]
        model = load_recommender("NU", hidden_dim=hdim, recommender_path=path,
                                 recommender_name="MLP", device=device,
                                 items_array=items_array, all_items_tensor=all_items_tensor,
                                 num_items=num_items)

    # -------- Evaluate per user --------
    rows = []
    users_used = 0
    for u in idx_users:
        train_u = train_array[u]
        test_u = test_array[u]

        # relevant = test positives
        relevant = set(np.where(test_u > 0)[0].tolist())
        if len(relevant) == 0:
            continue  # skip users with empty test set

        # build tensors/masks
        user_vec = torch.tensor(train_u, dtype=torch.float32, device=device)
        seen_mask = (train_u > 0)  # exclude these from recommendation

        ranked_ids, _ = rank_all_items(user_vec, model, items_array, seen_mask, device)

        # collect metrics for all Ks
        row = {"user_id": int(u), "num_train": int(train_u.sum()), "num_test": int(len(relevant))}
        for K in ks:
            row[f"HR@{K}"]   = hit_rate_at_k(ranked_ids, relevant, K)
            row[f"Prec@{K}"] = precision_at_k(ranked_ids, relevant, K)
            row[f"Rec@{K}"]  = recall_at_k(ranked_ids, relevant, K)
            row[f"NDCG@{K}"] = ndcg_at_k(ranked_ids, relevant, K)
            row[f"MAP@{K}"]  = apk(ranked_ids, relevant, K)
            row[f"MRR@{K}"]  = mrr_at_k(ranked_ids, relevant, K)
        rows.append(row)
        users_used += 1

    if users_used == 0:
        print("[WARN] No users with non-empty test sets after filtering.")
        return

    df = pd.DataFrame(rows)

    # -------- Aggregate & save --------
    summary = []
    for K in ks:
        block = {
            "K": K,
            "Users": users_used,
            "HR@K":   df[f"HR@{K}"].mean(),
            "Prec@K": df[f"Prec@{K}"].mean(),
            "Rec@K":  df[f"Rec@{K}"].mean(),
            "NDCG@K": df[f"NDCG@{K}"].mean(),
            "MAP@K":  df[f"MAP@{K}"].mean(),
            "MRR@K":  df[f"MRR@{K}"].mean(),
        }
        summary.append(block)

    summary_df = pd.DataFrame(summary)

    per_user_path = outdir / f"per_user_{args.dataset}_{args.recommender}.csv"
    summary_path  = outdir / f"summary_{args.dataset}_{args.recommender}.csv"
    df.to_csv(per_user_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Saved per-user metrics: {per_user_path}")
    print(f"[OK] Saved summary metrics:  {summary_path}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
