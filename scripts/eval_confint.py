# scripts/eval_confint.py
import argparse, pickle
import numpy as np
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import eval_one_expl_type
from src.utils import load_recommender
from src.config import recommender_path_dict, hidden_dim_dict
from src.nu_data import load_nu_dataset
from src.data_processing import load_and_preprocess_data

def bootstrap_ci(user_metric_matrix, B=1000, alpha=0.05, rng_seed=123):
    """
    user_metric_matrix: (U, S) where U users, S steps (Ke)
    returns: mean (S,), lo (S,), hi (S,)
    """
    U, S = user_metric_matrix.shape
    rng = np.random.default_rng(rng_seed)
    means = np.empty((B, S), dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, U, size=U)
        means[b] = user_metric_matrix[idx].mean(axis=0)
    lo = np.percentile(means, 100*alpha/2, axis=0)
    hi = np.percentile(means, 100*(1-alpha/2), axis=0)
    return user_metric_matrix.mean(axis=0), lo, hi

def collect_per_user_curves(
    expl_name, data_name, rec_name,
    test_array, test_data, items_array,
    recommender, kw, files_path, steps=5
):
    """
    Returns dict of 4 metrics, each as (U, steps): DEL, INS, NDCG, POS_at_20
    """
    files_path = Path(files_path)
    # We reuse eval_one_expl_type internals but here we capture per-user results.
    # To avoid editing src/metrics.py, we copy a slim loop here.
    import torch
    from tqdm import tqdm
    from src.metrics import single_user_metrics
    from src.utils import get_user_recommended_item

    # Load dict for dict-based explainers
    expl_dict = None
    if expl_name.lower() in ["lime","accent","lxr","shap"]:
        with open(files_path / f"{rec_name}_explanation_dicts.pkl","rb") as f:
            all_dicts = pickle.load(f)
        expl_dict = all_dicts.get(expl_name.lower())
        if expl_dict is None:
            raise ValueError(f"No {expl_name} in explanation dicts")

    U = test_array.shape[0]
    mats = {k: np.zeros((U, steps), dtype=np.float32)
            for k in ("DEL","INS","NDCG","POS_at_20")}

    with torch.no_grad():
        for u in tqdm(range(U), desc=f"{expl_name} per-user"):
            user_vector = test_array[u]
            user_tensor = torch.FloatTensor(user_vector).to(kw["device"])
            user_id = int(test_data.index[u]) if (test_data is not None and hasattr(test_data, "index")) else u

            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw).detach().cpu().numpy())
            item_tensor = torch.FloatTensor(items_array[item_id]).to(kw["device"])

            # zero target in history
            user_vector[item_id] = 0.0
            user_tensor[item_id] = 0.0

            if expl_name.lower() in ["lime","accent","lxr","shap"]:
                user_expl = expl_dict.get(user_id, expl_dict.get(u))
                if user_expl is None:
                    continue
            else:
                # on-the-fly baselines
                from src.metrics import single_user_expl
                user_expl = single_user_expl(
                    user_vector, user_tensor, item_id, item_tensor,
                    recommender, kw,
                    kw.get("pop_array"),
                    kw.get("jaccard_dict", {}),
                    kw.get("cosine_dict", {}),
                    None, mask_type=expl_name, user_id=user_id
                )

            DEL, INS, NDCG, POS = single_user_metrics(
                user_vector, user_tensor, item_id, item_tensor,
                recommender, user_expl,
                metric_type="discrete", steps=steps, mask_by="history", **kw
            )
            mats["DEL"][u] = DEL
            mats["INS"][u] = INS
            mats["NDCG"][u] = NDCG
            mats["POS_at_20"][u] = POS

    return mats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NU")
    ap.add_argument("--recommender", default="COSINE")
    ap.add_argument("--emb_csv", type=str, default=None)
    ap.add_argument("--views_csv", type=str, default=None)
    ap.add_argument("--eval_users", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--methods", type=str, default="jaccard,cosine,lime,accent,lxr,shap")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap samples")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files_path = Path("data/processed", args.dataset)

    # Load data (same as evaluate.py)
    if args.dataset == "NU":
        (train_data, test_data, static_test_data, pop_array,
         train_array, test_array, items_array, all_items_tensor,
         _pop_dup, item_embs, id2idx, idx2id) = load_nu_dataset(
            args.emb_csv, args.views_csv, device
        )
        static_test_np = static_test_data.to_numpy() if hasattr(static_test_data,"to_numpy") else static_test_data
    else:
        (train_data, test_data, static_test_data, pop_dict,
         train_array, test_array, items_array, all_items_tensor, pop_array) = load_and_preprocess_data(
            args.dataset, files_path, device
        )
        item_embs = None
        static_test_np = static_test_data if isinstance(static_test_data, np.ndarray) else static_test_data.to_numpy()

    if args.eval_users and args.eval_users > 0:
        n = min(args.eval_users, test_array.shape[0])
        test_array = test_array[:n]
        if hasattr(static_test_data, "iloc"):
            static_test_data = static_test_data.iloc[:n]
            static_test_np = static_test_data.to_numpy()
        elif isinstance(static_test_np, np.ndarray):
            static_test_np = static_test_np[:n]
        if hasattr(test_data, "iloc"):
            test_data = test_data.iloc[:n]

    num_items = items_array.shape[0]
    # normalized embedding matrix for cosine paths
    if args.dataset == "NU" and item_embs is not None:
        item_emb_matrix = np.asarray(item_embs, dtype=np.float32)
        norms = np.linalg.norm(item_emb_matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8
        item_emb_matrix = item_emb_matrix / norms
    else:
        item_emb_matrix = None

    from src.utils import load_recommender
    if args.recommender == "COSINE":
        recommender_path = None; hidden_dim = 0
    else:
        recommender_path = recommender_path_dict[(args.dataset, args.recommender)]
        hidden_dim = hidden_dim_dict[(args.dataset, args.recommender)]

    kw = {
        "device": device,
        "num_items": num_items,
        "num_features": num_items,
        "demographic": False,
        "pop_array": pop_array,
        "all_items_tensor": all_items_tensor,
        "static_test_data": static_test_np,
        "items_array": items_array,
        "recommender_name": args.recommender,
        "jaccard_dict": {},
        "item_embs": item_emb_matrix,
        "item_emb_matrix": item_emb_matrix,
        "k_masked": 1,
        "accent_topk": 5,
    }
    model = load_recommender(args.dataset, hidden_dim, recommender_path, **kw)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    out_dir = Path("results/ci"); out_dir.mkdir(parents=True, exist_ok=True)

    # Compute CIs and save CSVs + quick plots
    for m in methods:
        mats = collect_per_user_curves(m, args.dataset, args.recommender,
                                       test_array, test_data, items_array, model, kw,
                                       files_path, steps=args.steps)
        summary_rows = []
        for metric, M in mats.items():
            mean, lo, hi = bootstrap_ci(M, B=args.B)
            for ke in range(1, args.steps+1):
                summary_rows.append({
                    "method": m.upper(),
                    "metric": metric, "Ke": ke,
                    "mean": mean[ke-1], "lo": lo[ke-1], "hi": hi[ke-1],
                })

            # quick plot
            x = np.arange(1, args.steps+1)
            plt.figure(figsize=(7,5))
            plt.plot(x, mean, label=f"{m.upper()}")
            plt.fill_between(x, lo, hi, alpha=0.2)
            plt.xlabel("Kₑ"); plt.ylabel(metric)
            plt.title(f"{metric} with 95% CI — {m.upper()}")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(out_dir / f"{metric}_{m}_{args.dataset}_{args.recommender}.pdf")
            plt.close()

        df = pd.DataFrame(summary_rows)
        df.to_csv(out_dir / f"ci_{args.dataset}_{args.recommender}_{m}.csv", index=False)
        print(f"[OK] Saved CIs for {m} → {out_dir}")
        
if __name__ == "__main__":
    main()
