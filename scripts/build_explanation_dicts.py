# scripts/build_explanation_dicts.py
import argparse
from pathlib import Path
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# make src/ importable
sys.path.append(str(Path(__file__).parent.parent))

# Data loaders
from src.nu_data import load_nu_dataset
from src.data_processing import load_and_preprocess_data

# Recommender + helpers
from src.utils import load_recommender, get_user_recommended_item, recommender_run

# Explainers
from src.explainers import find_lime_mask, find_accent_mask

# LXR checkpoint config (absolute dir via config)
from src.config import LXR_checkpoint_dict, checkpoints_path


# -----------------------
# Helpers
# -----------------------
def to_dense_mask(imp, num_items: int) -> np.ndarray:
    """
    Normalize explainer outputs to a dense np.float32 vector of length num_items.
    Preserves float32 (no fp16 / no list casting).
    """
    if isinstance(imp, dict):
        vec = np.zeros(num_items, dtype=np.float32)
        for i, v in imp.items():
            try:
                ii = int(i)
            except Exception:
                continue
            if 0 <= ii < num_items:
                vec[ii] = float(v)
        return vec

    if isinstance(imp, (list, tuple)):
        vec = np.zeros(num_items, dtype=np.float32)
        for pair in imp:
            try:
                idx, score = pair
                ii = int(idx)
            except Exception:
                continue
            if 0 <= ii < num_items:
                vec[ii] = float(score)
        return vec

    arr = np.asarray(imp, dtype=np.float32).reshape(-1)
    if arr.shape[0] != num_items:
        arr = np.resize(arr, num_items).astype(np.float32, copy=False)
    return arr


def occlusion_importance(user_tensor, item_tensor, item_id, recommender, kw_dict) -> np.ndarray:
    """
    Fallback: Importance(i) = score(user) - score(user\\{i}) for seen items.
    Returns a dense np.float32 vector of length num_items.
    NOTE: No per-user max-normalization (preserve magnitude differences).
    """
    num_items = kw_dict["num_items"]
    kw = dict(kw_dict)
    kw.pop("output_type", None)

    with torch.no_grad():
        base = recommender_run(user_tensor, recommender, item_tensor, item_id,
                               output_type="single", **kw)
        base = float(base.detach().cpu().numpy())

    uv = user_tensor.detach().clone()
    seen_idx = torch.nonzero(uv > 0, as_tuple=False).view(-1).tolist()

    imp = np.zeros(num_items, dtype=np.float32)
    for i in seen_idx:
        uv_i = uv.clone()
        uv_i[i] = 0.0
        with torch.no_grad():
            s = recommender_run(uv_i, recommender, item_tensor, item_id,
                                output_type="single", **kw)
            s = float(s.detach().cpu().numpy())
        # keep non-negative drops
        imp[i] = max(0.0, base - s)

    # No normalization here (keep float32 magnitudes)
    return imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NU", choices=["NU", "ML1M", "Yahoo", "Pinterest"])
    ap.add_argument("--recommender", default="COSINE", choices=["COSINE", "MLP", "NCF", "VAE"])

    # NU only
    ap.add_argument("--emb_csv", type=str, default=None, help="NU only: embeddings CSV")
    ap.add_argument("--views_csv", type=str, default=None, help="NU only: views CSV")

    # Generic split params (for NU they are forwarded to load_nu_dataset)
    ap.add_argument("--min_profile", type=int, default=5)
    ap.add_argument("--min_test", type=int, default=5)
    ap.add_argument("--split_ratio", type=float, default=0.8)

    # User selection
    ap.add_argument("--eval_users", type=int, default=0, help="Limit to N users (0 = all)")
    ap.add_argument("--random_eval", action="store_true", help="Randomly sample eval users")
    ap.add_argument("--seed", type=int, default=42, help="Seed for random_eval")

    # Methods
    ap.add_argument("--methods", type=str, default="lime,accent,lxr,shap",
                    help="Comma list from: lime,accent,lxr,shap")
    ap.add_argument("--accent_topk", type=int, default=5)

    # Logging / checkpointing
    ap.add_argument("--save_every", type=int, default=1000, help="Save partial dict every N users")
    ap.add_argument("--log_every", type=int, default=100, help="Print a short log every N users")

    args = ap.parse_args()

    data_name = args.dataset
    rec_name = args.recommender
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files_path = Path("data/processed", data_name)
    files_path.mkdir(parents=True, exist_ok=True)

    # ===== Load data =====
    if data_name == "NU":
        (train_data, test_data, static_test_data, pop_array,
         train_array, test_array, items_array, all_items_tensor,
         _pop_dup, item_embs, id2idx, idx2id) = load_nu_dataset(
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

    # ===== Optionally limit users (random or first-N) =====
    if args.eval_users and args.eval_users > 0:
        n_all = test_array.shape[0]
        n = min(args.eval_users, n_all)
        if args.random_eval:
            rng = np.random.default_rng(args.seed)
            sel = rng.choice(n_all, size=n, replace=False)
            sel = np.sort(sel)  # keep stable order
            test_array = test_array[sel]
            if isinstance(static_test_data, pd.DataFrame):
                static_test_data = static_test_data.iloc[sel]
                static_test_np = static_test_data.to_numpy()
            else:
                static_test_np = static_test_np[sel]
            if isinstance(test_data, pd.DataFrame):
                test_data = test_data.iloc[sel]
        else:
            test_array = test_array[:n]
            if isinstance(static_test_data, pd.DataFrame):
                static_test_data = static_test_data.iloc[:n]
                static_test_np = static_test_data.to_numpy()
            else:
                static_test_np = static_test_np[:n]
            if isinstance(test_data, pd.DataFrame):
                test_data = test_data.iloc[:n]

    num_items = items_array.shape[0]

    # ===== Build/normalize embeddings for cosine methods (if available) =====
    if data_name == "NU" and item_embs is not None:
        item_emb_matrix = np.asarray(item_embs, dtype=np.float32)
        norms = np.linalg.norm(item_emb_matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1e-8
        item_emb_matrix = item_emb_matrix / norms
    else:
        item_emb_matrix = None

    # ===== Shared kwargs passed into explainers/recommender =====
    kw = {
        "device": device,
        "num_items": num_items,
        "num_features": num_items,
        "demographic": False,
        "pop_array": pop_array,
        "all_items_tensor": all_items_tensor,
        "static_test_data": static_test_np,
        "items_array": items_array,
        "recommender_name": rec_name,
        # LIME config (tabular LIME over binary masks)
        "min_pert": 50,
        "max_pert": 100,
        "num_of_perturbations": 150,
        # Cosine plumbing
        "item_embs": item_emb_matrix,
        "item_emb_matrix": item_emb_matrix,
        "k_masked": 1,
        "accent_topk": args.accent_topk,
    }
    kw_lime = dict(kw); kw_lime["output_type"] = "vector"  # LIME expects vector scoring
    kw_no_ot = dict(kw)  # ACCENT/LXR/SHAP must not get output_type twice

    # ===== Recommender =====
    if rec_name == "COSINE":
        recommender_path = None
        hidden_dim = 0
    else:
        from src.config import recommender_path_dict, hidden_dim_dict
        recommender_path = recommender_path_dict[(data_name, rec_name)]
        hidden_dim = hidden_dim_dict[(data_name, rec_name)]
    recommender = load_recommender(data_name, hidden_dim, recommender_path, **kw)

    # ===== Optional: LXR explainer =====
    need_lxr = "lxr" in methods
    lxr_explainer = None
    if need_lxr:
        try:
            from src.explainers import load_explainer
            lxr_explainer = load_explainer(
                LXR_checkpoint_dict, data_name, rec_name,
                checkpoints_path=checkpoints_path,
                num_items=num_items, num_features=num_items, device=device
            )
        except Exception as e:
            print(f"[WARN] LXR unavailable: {e}. Skipping LXR.")
            methods = [m for m in methods if m != "lxr"]

    # ===== Optional: SHAP artifacts =====
    need_shap = "shap" in methods
    shap_values = None
    item_to_cluster = None
    if need_shap:
        try:
            shap_values = np.load(files_path / "shap_values.npy")
            item_to_cluster = np.load(files_path / "item_to_cluster.npy")
            if item_to_cluster.shape[0] != num_items:
                raise ValueError(f"item_to_cluster length {item_to_cluster.shape[0]} != num_items {num_items}")
        except Exception as e:
            print(f"[WARN] SHAP artifacts missing or invalid: {e}. Skipping SHAP.")
            methods = [m for m in methods if m != "shap"]

    # ===== Load existing dict (merge) so we don’t overwrite other methods =====
    save_path = files_path / f"{rec_name}_explanation_dicts.pkl"
    if save_path.exists():
        try:
            with open(save_path, "rb") as f:
                out = pickle.load(f)
            if not isinstance(out, dict):
                out = {}
        except Exception:
            out = {}
    else:
        out = {}
    for m in methods:
        out.setdefault(m, {})

    # ===== Resume from partial if present =====
    resume_path = files_path / f"{rec_name}_explanation_dicts.partial.pkl"
    if resume_path.exists():
        try:
            prev = pickle.loads(resume_path.read_bytes())
            for m in methods:
                if isinstance(prev.get(m), dict):
                    out[m].update(prev[m])
            print(f"[RESUME] Loaded partial explanations from {resume_path}", flush=True)
        except Exception as e:
            print(f"[WARN] Could not resume from partial file: {e}", flush=True)

    SAVE_EVERY = max(1, int(args.save_every))
    LOG_EVERY = max(1, int(args.log_every))

    # ===== Iterate users and compute per-user masks =====
    with torch.no_grad():
        pbar = tqdm(range(test_array.shape[0]), desc="Users", mininterval=1.0)
        for idx in pbar:
            user_vector = test_array[idx]
            user_tensor = torch.FloatTensor(user_vector).to(device)

            # prefer stable external user id if available
            if isinstance(test_data, pd.DataFrame) and hasattr(test_data, "index"):
                try:
                    user_id = int(test_data.index[idx])
                except Exception:
                    user_id = int(idx)
            else:
                user_id = int(idx)

            # top-1 item to explain
            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw).detach().cpu().numpy())
            item_tensor = torch.FloatTensor(items_array[item_id]).to(device)

            # zero the target in the history (align with metrics masking)
            user_vector[item_id] = 0.0
            user_tensor[item_id] = 0.0
            has_hist = int((user_tensor > 0).sum().item()) > 0

            # ----- LIME -----
            if "lime" in methods:
                if not has_hist:
                    out["lime"][user_id] = np.zeros(num_items, dtype=np.float32)
                else:
                    try:
                        imp = find_lime_mask(user_tensor, item_id, recommender, item_tensor, **kw_lime)
                        out["lime"][user_id] = to_dense_mask(imp, num_items)
                    except Exception as e:
                        print(f"[WARN] LIME failed for user {user_id}: {e} — using occlusion fallback", flush=True)
                        out["lime"][user_id] = occlusion_importance(user_tensor, item_tensor, item_id, recommender, kw)

            # ----- ACCENT -----
            if "accent" in methods:
                try:
                    imp = find_accent_mask(user_tensor, user_id, item_tensor, item_id,
                                           recommender, args.accent_topk, **kw_no_ot)
                    out["accent"][user_id] = to_dense_mask(imp, num_items)
                except Exception as e:
                    print(f"[WARN] ACCENT failed for user {user_id}: {e}", flush=True)

            # ----- LXR -----
            if "lxr" in methods and lxr_explainer is not None:
                try:
                    from src.explainers import find_lxr_mask
                    imp = find_lxr_mask(user_tensor, item_tensor, lxr_explainer, **kw_no_ot)
                    out["lxr"][user_id] = to_dense_mask(imp, num_items)
                except Exception as e:
                    print(f"[WARN] LXR failed for user {user_id}: {e}", flush=True)

            # ----- SHAP -----
            if "shap" in methods and shap_values is not None and item_to_cluster is not None:
                try:
                    from src.explainers import find_shapley_mask
                    imp = find_shapley_mask(user_tensor, user_id, recommender,
                                            shap_values, item_to_cluster, **kw_no_ot)
                    out["shap"][user_id] = to_dense_mask(imp, num_items)
                except Exception as e:
                    print(f"[WARN] SHAP failed for user {user_id}: {e}", flush=True)

            # periodic log (lightweight)
            if (idx + 1) % LOG_EVERY == 0:
                sizes = {m: len(out.get(m, {})) for m in methods}
                print(f"[{idx+1}/{test_array.shape[0]}] users processed -> sizes {sizes}", flush=True)

            # periodic checkpoint (only the methods you’re computing)
            if (idx + 1) % SAVE_EVERY == 0:
                tmp = {m: out.get(m, {}) for m in methods}
                with open(resume_path, "wb") as f:
                    pickle.dump(tmp, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[CKPT] Saved partial at user {idx+1} -> {resume_path}", flush=True)

    # ===== Save combined dicts =====
    with open(save_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    if resume_path.exists():
        try:
            resume_path.unlink()
        except Exception:
            pass
    print(f"[OK] Saved explanation dicts: {save_path}", flush=True)


if __name__ == "__main__":
    main()
