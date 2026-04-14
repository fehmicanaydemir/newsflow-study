# scripts/train_lxr.py
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# make src/ importable
sys.path.append(str(Path(__file__).parent.parent))

from src.nu_data import load_nu_dataset
from src.data_processing import load_and_preprocess_data
from src.explainers import Explainer
from src.utils import load_recommender, sample_indices


def one_hot_from_index(items_array: np.ndarray, idx: int) -> torch.Tensor:
    """Return 1xI one-hot item vector as torch.FloatTensor on CPU (caller moves to device)."""
    vec = items_array[int(idx)]
    return torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)  # shape (1, I)


def train_epoch(
    explainer: nn.Module,
    recommender: nn.Module,
    train_mat: np.ndarray,         # shape [U, I+2] from utils.sample_indices
    items_array: np.ndarray,       # (I, I) one-hots
    device: torch.device,
    l1_lambda: float = 1e-3,
    bpr_weight: float = 1.0,
) -> float:
    """
    One pass over all users (simple per-sample training).
    Loss = BPR(s_pos, s_neg) + l1_lambda * ||mask_seen||_1
    """
    explainer.train()
    total_loss = 0.0

    I = items_array.shape[0]
    for row in tqdm(train_mat, desc="train"):
        user_vec = row[:-2].astype(np.float32)  # masked feature vector (pos masked out)
        pos_idx = int(row[-2])
        neg_idx = int(row[-1])

        # to torch
        user = torch.from_numpy(user_vec).unsqueeze(0).to(device)        # (1, I)
        pos_item = one_hot_from_index(items_array, pos_idx).to(device)    # (1, I)
        neg_item = one_hot_from_index(items_array, neg_idx).to(device)    # (1, I)

        # forward -> masks (in [0,1])
        mask_pos = explainer(user, pos_item)  # (1, I)
        mask_neg = explainer(user, neg_item)  # (1, I)

        # apply mask to user history
        user_pos = user * mask_pos
        user_neg = user * mask_neg

        # get scores from your recommender
        s_pos = recommender(user_pos, pos_item).squeeze()
        s_neg = recommender(user_neg, neg_item).squeeze()

        # BPR loss
        bpr = torch.nn.functional.softplus(-(s_pos - s_neg))  # log(1+exp(-(pos-neg)))
        # L1 over mask on seen positions (user>0)
        seen = (user > 0).float()
        l1 = ((mask_pos * seen).abs().sum() + (mask_neg * seen).abs().sum()) / (seen.sum() + 1e-6)

        loss = bpr_weight * bpr + l1_lambda * l1
        loss.backward()
        total_loss += float(loss.item())

        # single-sample step (simple & robust)
        for p in explainer.parameters():
            if p.grad is not None and torch.isfinite(p.grad).all():
                pass  # grads ok
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, train_mat.shape[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NU", choices=["NU", "ML1M", "Yahoo", "Pinterest"])
    ap.add_argument("--recommender", default="COSINE", choices=["COSINE","MLP","NCF","VAE"])

    # NU only
    ap.add_argument("--emb_csv", type=str, default=None, help="NU only")
    ap.add_argument("--views_csv", type=str, default=None, help="NU only")
    ap.add_argument("--min_profile", type=int, default=5)
    ap.add_argument("--min_test", type=int, default=5)
    ap.add_argument("--split_ratio", type=float, default=0.8)

    # Training hparams
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1_lambda", type=float, default=1e-3)
    ap.add_argument("--bpr_weight", type=float, default=1.0)

    # Output
    ap.add_argument("--out", type=str, default=None, help="Output filename under checkpoints/")
    args = ap.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # where to save
    from src.config import CHECKPOINTS_ROOT
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        # name by dataset & recommender for convenience
        args.out = f"lxr_explainer_{args.dataset}_{args.recommender}.pt"
    out_path = CHECKPOINTS_ROOT / args.out

    # ===== Load data (mirrors evaluate.py) =====
    data_name = args.dataset
    rec_name = args.recommender

    if data_name == "NU":
        (train_data, test_data, static_test_data, pop_array,
         train_array, test_array, items_array, all_items_tensor,
         _pop_dup, item_embs, id2idx, idx2id) = load_nu_dataset(
            args.emb_csv, args.views_csv, device,
            min_profile=args.min_profile, min_test=args.min_test, split_ratio=args.split_ratio
        )
        item_emb_matrix = None
        if item_embs is not None:
            item_emb_matrix = np.asarray(item_embs, dtype=np.float32)
            norms = np.linalg.norm(item_emb_matrix, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1e-8
            item_emb_matrix = item_emb_matrix / norms
    else:
        (train_data, test_data, static_test_data, pop_dict,
         train_array, test_array, items_array, all_items_tensor, pop_array) = load_and_preprocess_data(
            data_name, Path("data/processed", data_name), device
        )
        item_emb_matrix = None  # (provide if you want COSINE for non-NU)

    num_items = items_array.shape[0]

    # ===== Build recommender (reused for scoring) =====
    kw_dict = {
        "device": device,
        "num_items": num_items,
        "num_features": num_items,
        "demographic": False,
        "pop_array": pop_array,
        "all_items_tensor": all_items_tensor,
        "static_test_data": static_test_data if data_name != "NU" else (static_test_data.to_numpy() if hasattr(static_test_data, "to_numpy") else static_test_data),
        "items_array": items_array,
        "recommender_name": rec_name,
        # cosine plumbing
        "item_embs": item_emb_matrix,
        "item_emb_matrix": item_emb_matrix,
    }

    # For MLP/NCF/VAE this would load checkpoints from config; COSINE uses embeddings
    recommender = load_recommender(data_name, hidden_dim=0, recommender_path=None, **kw_dict)
    recommender.eval()
    for p in recommender.parameters():
        p.requires_grad = False

    # ===== Build explainer =====
    explainer = Explainer(user_size=num_items, item_size=num_items, hidden_size=args.hidden_dim, device=device).to(device)
    global optimizer
    optimizer = optim.Adam(explainer.parameters(), lr=args.lr)

    # ===== Build training matrix with pos/neg indices =====
    # Each row: [user_vector(with pos masked), pos_idx, neg_idx]
    train_mat = sample_indices(train_array, pop_array=pop_array).astype(np.float32)

    print(f"[INFO] Training LXR on {train_mat.shape[0]} users; items={num_items}")
    print(f"[INFO] Saving to: {out_path}")

    # ===== Train =====
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            explainer, recommender, train_mat, items_array, device,
            l1_lambda=args.l1_lambda, bpr_weight=args.bpr_weight
        )
        print(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.4f}")

    # ===== Save explainer checkpoint =====
    torch.save(explainer.state_dict(), out_path)
    print(f"[OK] Saved LXR explainer: {out_path}")
