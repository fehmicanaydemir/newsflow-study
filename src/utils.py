# src/utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn

# ============================================================
# New: user-embedding cosine helpers (no item–item dependency)
# ============================================================
def _safe_l2_norm(x, eps: float = 1e-8):
    if isinstance(x, np.ndarray):
        n = np.linalg.norm(x, ord=2)
        return max(float(n), eps)
    else:
        return torch.clamp(torch.linalg.norm(x, ord=2), min=eps)

def compute_user_embedding(
    user_vec: np.ndarray | torch.Tensor,
    item_emb_matrix: np.ndarray | torch.Tensor,
    eps: float = 1e-8,
):
    """
    Differentiable user embedding = weighted mean of item embeddings.
    Accepts 1D [I] or 2D [1, I] user vectors. Returns 1D [D].
    For binary histories this collapses to the simple mean over seen items.
    """
    if isinstance(user_vec, torch.Tensor):
        # accept batch-of-1
        if user_vec.dim() == 2 and user_vec.size(0) == 1:
            user_vec = user_vec.squeeze(0)
        w = user_vec.to(item_emb_matrix.dtype)                   # [I]
        sum_w = torch.clamp(w.sum(), min=eps)                    # scalar
        # item_emb_matrix: [I, D] --> u: [D]
        u = (item_emb_matrix.t() @ w) / sum_w
        return u
    else:
        uv = np.asarray(user_vec)
        if uv.ndim == 2 and uv.shape[0] == 1:
            uv = uv.squeeze(0)
        w = uv.astype(item_emb_matrix.dtype, copy=False)         # [I]
        sum_w = float(w.sum())
        if not np.isfinite(sum_w) or sum_w <= eps:
            return np.zeros(item_emb_matrix.shape[1], dtype=item_emb_matrix.dtype)
        u = (item_emb_matrix.T @ w) / sum_w                      # [D]
        return u




def cosine_scores_from_user(
    user_vec: np.ndarray | torch.Tensor,
    item_emb_matrix: np.ndarray | torch.Tensor,
    candidate_mask: Optional[np.ndarray | torch.Tensor] = None
):
    """
    Cosine similarity between the user embedding and all item embeddings.
    Coerces item_emb_matrix (and mask) to match user_vec backend.
    """
    if isinstance(user_vec, torch.Tensor):
        # to torch
        if not isinstance(item_emb_matrix, torch.Tensor):
            item_emb_matrix = torch.as_tensor(item_emb_matrix, dtype=torch.float32, device=user_vec.device)
        if candidate_mask is not None and not isinstance(candidate_mask, torch.Tensor):
            candidate_mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=user_vec.device)

        u = compute_user_embedding(user_vec, item_emb_matrix)
        u_norm = u / _safe_l2_norm(u)
        denom = torch.clamp(torch.linalg.norm(item_emb_matrix, dim=1, keepdim=True), min=1e-8)
        items_norm = item_emb_matrix / denom
        scores = items_norm @ u_norm  # [num_items]
        if candidate_mask is not None:
            scores = scores.clone()
            scores[~candidate_mask] = float("-inf")
        return scores

    else:
        # to numpy
        if isinstance(item_emb_matrix, torch.Tensor):
            item_emb_matrix = item_emb_matrix.detach().cpu().numpy()
        if candidate_mask is not None and isinstance(candidate_mask, torch.Tensor):
            candidate_mask = candidate_mask.detach().cpu().numpy().astype(bool)

        u = compute_user_embedding(user_vec, item_emb_matrix)
        u_norm = u / _safe_l2_norm(u)
        denom = np.linalg.norm(item_emb_matrix, axis=1, keepdims=True)
        denom[denom < 1e-8] = 1e-8
        items_norm = item_emb_matrix / denom
        scores = items_norm @ u_norm  # [num_items]
        if candidate_mask is not None:
            scores = scores.copy()
            scores[~candidate_mask] = -np.inf
        return scores


def cosine_rank_all(
    user_vec: np.ndarray | torch.Tensor,
    item_emb_matrix: np.ndarray | torch.Tensor,
    exclude_seen: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (indices_sorted_desc, scores_sorted_desc) for ALL items.
    If exclude_seen=True, seen items (user_vec>0) are set to -inf before sorting.
    """
    if isinstance(user_vec, torch.Tensor):
        candidate_mask = (user_vec == 0) if exclude_seen else None
        scores = cosine_scores_from_user(user_vec, item_emb_matrix, candidate_mask=candidate_mask)
        s = scores.detach().cpu().numpy()
    else:
        candidate_mask = (user_vec == 0) if exclude_seen else None
        s = cosine_scores_from_user(user_vec, item_emb_matrix, candidate_mask=candidate_mask)

    order = np.argsort(-s)
    return order, s[order]

def cosine_top_k(
    user_vec: np.ndarray | torch.Tensor,
    item_emb_matrix: np.ndarray | torch.Tensor,
    k: int = 5,
    exclude_seen: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-k unseen items by cosine similarity to the user embedding.
    Returns (topk_indices, topk_scores) sorted descending by score.
    """
    idx_all, s_all = cosine_rank_all(user_vec, item_emb_matrix, exclude_seen=exclude_seen)
    k = int(k)
    k = max(0, min(k, idx_all.size))
    return idx_all[:k], s_all[:k]


# ---------------------------
# Negative sampling (training)
# ---------------------------
def sample_indices(data_copy, **kw) -> np.ndarray:
    """
    Returns a numpy array of shape [U, I+2]:
      - first I columns: user vector (with one positive masked out to avoid leakage)
      - last 2 columns: (pos_idx, neg_idx) as float32
    Robust to NaNs/zeros in popularity.
    """
    if not isinstance(data_copy, pd.DataFrame):
        data_copy = pd.DataFrame(data_copy)

    num_users, num_items = data_copy.shape

    pop_array = kw.get("pop_array", None)
    if pop_array is None:
        pop_array = np.ones(num_items, dtype=np.float64)
    else:
        pop_array = np.asarray(pop_array, dtype=np.float64)
        if pop_array.shape[0] != num_items:
            pop_array = np.resize(pop_array, num_items)

    mat = data_copy.to_numpy(dtype=np.float32, copy=True)
    pos_idx_all = np.full(num_users, -1, dtype=np.int64)
    neg_idx_all = np.full(num_users, -1, dtype=np.int64)

    rng = np.random.default_rng()

    for u in range(num_users):
        row = mat[u]
        ones = np.flatnonzero(row > 0)
        zeros = np.flatnonzero(row <= 0)

        # choose a positive
        if ones.size == 0:
            pos_i = int(np.argmax(pop_array))
        else:
            pos_i = int(ones[rng.integers(0, ones.size)])
        pos_idx_all[u] = pos_i

        # mask that positive in the feature vector
        mat[u, pos_i] = 0.0

        # choose a negative
        if zeros.size == 0:
            neg_i = pos_i
        else:
            probs = np.asarray(pop_array[zeros], dtype=np.float64)
            probs = np.where(np.isfinite(probs), probs, 0.0)
            probs = np.maximum(probs, 0.0)
            s = probs.sum()
            if s > 0:
                probs = probs / s
            else:
                probs = np.full(zeros.size, 1.0 / zeros.size, dtype=np.float64)

            s2 = probs.sum()
            if not np.isfinite(s2) or s2 <= 0.0:
                probs = np.full(zeros.size, 1.0 / zeros.size, dtype=np.float64)
            else:
                probs = probs / s2

            neg_i = int(zeros[rng.choice(zeros.size, p=probs)])
        neg_idx_all[u] = neg_i

    out = np.concatenate(
        [mat, pos_idx_all[:, None].astype(np.float32), neg_idx_all[:, None].astype(np.float32)],
        axis=1
    )
    return out


# -----------------------------------
# Unified scorer for recommenders (MLP)
# -----------------------------------
def recommender_run(
    user_tensor: torch.Tensor,
    recommender: nn.Module,
    item_tensor: Optional[torch.Tensor] = None,
    item_id: Optional[int] = None,
    output_type: str = "vector",
    **kw: Any,
) -> torch.Tensor:
    """
    - output_type == 'vector': returns scores for ALL items
    - output_type == 'single': returns score for ONE item (via item_id or item_tensor)
    MLP forward expects (user_tensor, item_tensor).
    """
    device = kw.get("device", user_tensor.device)

    if output_type == "vector":
        if item_tensor is None:
            all_items = kw.get("all_items_tensor", None)
            if all_items is None:
                items_array = kw.get("items_array", None)
                if items_array is None:
                    raise ValueError("MLP vector scoring needs 'all_items_tensor' or 'items_array'.")
                all_items = torch.tensor(items_array, dtype=torch.float32, device=device)
            item_tensor = all_items
        return recommender(user_tensor, item_tensor).squeeze()

    elif output_type == "single":
        if item_tensor is None:
            if item_id is None:
                raise ValueError("single-item scoring requires item_id or item_tensor")
            items_array = kw["items_array"]
            vec = items_array[int(item_id)]
            item_tensor = torch.tensor(vec, dtype=torch.float32, device=device)
        return recommender(user_tensor, item_tensor).squeeze()

    else:
        raise ValueError(f"Unknown output_type: {output_type}")


@torch.no_grad()
def get_user_recommended_item(user_tensor, recommender, **kw):
    """
    Return the top-1 recommended item index for a user (MLP path).
    Always gets the vector of scores first; masks out seen items if needed upstream.
    """
    num_items = kw['num_items']
    kw = dict(kw)
    kw.pop('output_type', None)

    scores = recommender_run(user_tensor, recommender, output_type='vector', **kw)
    scores = scores[:num_items]
    return torch.argmax(scores)


def get_top_k(user_tensor: torch.Tensor, original_user_tensor: torch.Tensor, model: nn.Module, **kw: Any) -> Dict[int, float]:
    """
    Rank ALL items by the model score for `user_tensor`.
    If kw['exclude_seen_eval'] is True, items that are 1 in `original_user_tensor`
    are suppressed; otherwise we keep them (paper-style global evaluation).
    Returns a dict {item_id: score} in descending score order.
    """
    num_items = kw["num_items"]
    # avoid passing output_type twice
    kw_local = dict(kw)
    kw_local.pop("output_type", None)

    exclude_seen = kw_local.pop("exclude_seen_eval", True)  # <— key flag

    scores = recommender_run(user_tensor, model, output_type="vector", **kw_local)[:num_items]
    scores_np = scores.detach().cpu().numpy()

    if exclude_seen:
        seen = (original_user_tensor.detach().cpu().numpy()[:num_items] > 0.0)
        scores_np = np.where(seen, -1e9, scores_np)

    idx = np.argsort(-scores_np)  # descending
    return {int(i): float(scores_np[i]) for i in idx if scores_np[i] > -1e8}




def get_index_in_the_list(
    user_tensor: torch.Tensor,
    original_user_tensor: torch.Tensor,
    item_id: int,
    recommender: nn.Module,
    **kw: Any
) -> int:
    ordered = list(get_top_k(user_tensor, original_user_tensor, recommender, **kw).keys())
    return ordered.index(item_id) if item_id in ordered else len(ordered)


def get_ndcg(ranked_list: list, target_item: int, **kw: Any) -> float:
    """
    ranked_list: list of item ids in descending score order (e.g., list(dict.keys()))
    """
    if target_item not in ranked_list:
        return 0.0
    pos = ranked_list.index(target_item)
    # DCG with log2 rank discount; +2 because pos is 0-based
    return float(1.0 / np.log2(pos + 2))


# -----------------------------------
# Batched HR@10 evaluation (training)
# -----------------------------------
@torch.no_grad()
def recommender_evaluations(model: nn.Module, **kw: Any) -> Tuple[float, None, None, None, None]:
    """
    Proper HR@10 via candidate ranking (MLP path).
    static_test_data rows are [user_vector..., pos_idx, neg_idx] (last two may be float).
    """
    device = kw["device"]
    items_array = kw["items_array"]           # (I, I) one-hots
    static_test_data = kw["static_test_data"] # DataFrame or ndarray
    num_items = kw["num_items"]

    K_cand = 100   # 1 positive + 99 negatives
    TOPK = 10
    B = 128        # user batch size

    test_mat = static_test_data.to_numpy() if hasattr(static_test_data, "to_numpy") else static_test_data
    X_all = torch.as_tensor(test_mat[:, :-2], dtype=torch.float32, device=device)
    pos_raw = test_mat[:, -2]
    pos_idx = np.nan_to_num(pos_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
    pos_idx = np.clip(pos_idx, 0, num_items - 1)

    N = X_all.shape[0]
    rng = np.random.default_rng(123)
    I = np.arange(num_items, dtype=np.int64)

    hr_hits = 0
    model.eval()
    for start in range(0, N, B):
        end = min(N, start + B)
        X = X_all[start:end]                 # (b, I)
        b = X.shape[0]
        pos_idx_b = pos_idx[start:end]       # (b,)

        # sample negatives
        neg_idx_b = np.empty((b, K_cand - 1), dtype=np.int64)
        for i, p in enumerate(pos_idx_b):
            negs = rng.choice(I, size=K_cand - 1, replace=False)
            if (negs == p).any():
                mask = (negs == p)
                need = int(mask.sum())
                repl = rng.choice(I[I != p], size=need, replace=False)
                negs[mask] = repl
            neg_idx_b[i] = negs

        cand_idx = np.concatenate([pos_idx_b[:, None], neg_idx_b], axis=1)  # (b, K)
        cand_items_np = items_array[cand_idx]                                # (b, K, I)

        flat_items = torch.as_tensor(cand_items_np.reshape(-1, num_items),
                                     dtype=torch.float32, device=device)    # (b*K, I)
        X_rep = X.repeat_interleave(K_cand, dim=0)                           # (b*K, I)

        S = model(X_rep, flat_items)                                         # (b*K, b*K) or (b*K,)
        if S.dim() == 2:
            flat_scores = torch.diagonal(S)                                  # (b*K,)
        else:
            flat_scores = S.squeeze()
        scores = flat_scores.view(b, K_cand)                                 # (b, K)

        pos_scores = scores[:, 0]
        better = (scores[:, 1:] > pos_scores[:, None]).sum(dim=1)
        ranks = better + 1
        hr_hits += int((ranks <= TOPK).sum().item())

    hr_at_10 = hr_hits / float(N)
    return hr_at_10, None, None, None, None

# ---------- Minimal recommender loader to satisfy evaluate.py ----------

def load_recommender(data_name, hidden_dim, recommender_path, **kw_dict):
    """
    Build and load a recommender for evaluation.
    Supports MLP (trained checkpoint) and optional COSINE model wrapper.
    - For COSINE you can bypass model classes entirely and just use `cosine_top_k`
      with the raw item embedding matrix.
    """
    device = kw_dict.get("device", torch.device("cpu"))
    recommender_name = kw_dict.get("recommender_name", "MLP")

    if recommender_name == "MLP":
        # Lazy import to avoid circulars
        from src.models import MLP
        inferred_hidden = hidden_dim
        if recommender_path:
            state = torch.load(str(recommender_path), map_location=device)
            if "users_fc.weight" in state:
                inferred_hidden = int(state["users_fc.weight"].shape[0])
        model = MLP(inferred_hidden, **kw_dict).to(device)
        if recommender_path:
            state = torch.load(str(recommender_path), map_location=device)
            model.load_state_dict(state, strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    elif recommender_name == "COSINE":
        class _CosineWrapper(nn.Module):
            def __init__(self, item_embs: torch.Tensor):
                super().__init__()
                self.register_buffer("item_embs", item_embs)

            def forward(self, user_tensor: torch.Tensor, items_tensor: Optional[torch.Tensor] = None):
                # Full score vector: cosine(user-embedding, each item embedding)
                scores = cosine_scores_from_user(user_tensor, self.item_embs, candidate_mask=None)  # [I]

                # Return a scalar ONLY if a single item vector is provided (I,) or (1, I).
                if items_tensor is not None:
                    if items_tensor.dim() == 1 and items_tensor.shape[0] == scores.shape[0]:
                        return (scores * items_tensor).sum()
                    if items_tensor.dim() == 2 and items_tensor.shape[0] == 1 and items_tensor.shape[1] == scores.shape[0]:
                        return (scores * items_tensor[0]).sum()

                # Otherwise (e.g., items_tensor is all items [I, I]) return the full vector
                return scores



        item_embs = kw_dict.get("item_embs", None)
        if item_embs is None:
            raise ValueError("COSINE recommender requires 'item_embs' in kw_dict.")
        item_embs_t = item_embs if isinstance(item_embs, torch.Tensor) \
            else torch.tensor(item_embs, dtype=torch.float32, device=device)
        model = _CosineWrapper(item_embs_t).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    else:
        raise NotImplementedError(f"load_recommender: unsupported recommender '{recommender_name}'")
