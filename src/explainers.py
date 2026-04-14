# src/explainers.py
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# Make package imports work when running scripts/*
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import get_top_k, recommender_run
from src.lime import LimeBase, get_lime_args, get_lire_args, distance_to_proximity


__all__ = [
    "Explainer",
    "load_explainer",
    "find_pop_mask",
    "find_jaccard_mask",
    "find_cosine_mask",
    "find_lime_mask",
    "find_lire_mask",
    "find_lxr_mask",
    "find_fia_mask",
    "find_shapley_mask",
    "find_accent_mask",
]


# =========================
# LXR Explainer (neural)
# =========================
class Explainer(nn.Module):
    """
    Small bottleneck network used by LXR to output per-item relevance scores.
    Outputs a dense vector of length = user_size (== num_items) in [0,1].
    """
    def __init__(self, user_size, item_size, hidden_size, device):
        super().__init__()
        self.device = device
        self.users_fc = nn.Linear(in_features=user_size, out_features=hidden_size).to(device)
        self.items_fc = nn.Linear(in_features=item_size, out_features=hidden_size).to(device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size).to(device),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=user_size).to(device),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, user_tensor, item_tensor):
        u = self.users_fc(user_tensor.float())
        i = self.items_fc(item_tensor.float())
        x = torch.cat((u, i), dim=-1)
        scores = self.bottleneck(x).to(self.device)
        return scores


def load_explainer(LXR_checkpoint_dict, data_name, recommender_name, checkpoints_path, num_items, num_features, device):
    """
    Load a pretrained LXR explainer. Expects:
      LXR_checkpoint_dict[(data_name, recommender_name)] -> (checkpoint_file, hidden_dim)
    """
    lxr_path, lxr_dim = LXR_checkpoint_dict[(data_name, recommender_name)]
    explainer = Explainer(num_features, num_items, lxr_dim, device)
    ckpt = torch.load(Path(checkpoints_path, lxr_path), map_location="cpu")
    explainer.load_state_dict(ckpt)
    explainer.eval()
    for p in explainer.parameters():
        p.requires_grad = False
    return explainer


# =========================
# Simple baselines
# =========================
def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _clone_like(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    return np.array(x, copy=True)


def find_pop_mask(x, item_id, pop_array, **kw):
    """
    Popularity: importance = item popularity but only for items actually seen by the user.
    Returns dict {item_idx: score} over seen items.
    """
    uv = _to_np(x).astype(float)
    uv[item_id] = 0.0
    seen = np.where(uv > 0)[0]
    out = {}
    for i in seen:
        out[i] = float(pop_array[i])
    return out


def find_jaccard_mask(x, item_id, user_based_Jaccard_sim, **kw):
    """
    Jaccard: importance = Jaccard(sim) between seen item i and the target item.
    Returns dict {i: sim}.
    """
    uv = _to_np(x).astype(float)
    uv[item_id] = 0.0
    seen = np.where(uv > 0)[0]
    out = {}
    for i in seen:
        out[i] = float(
            user_based_Jaccard_sim.get((int(i), int(item_id)),
            user_based_Jaccard_sim.get((int(item_id), int(i)), 0.0))
        )
    return out


def find_cosine_mask(user_vec: np.ndarray | torch.Tensor,
                     item_emb_matrix: np.ndarray | torch.Tensor,
                     target_item_id: int,
                     k_masked: int = 1):
    """
    Returns a dict {history_item_id: importance} explaining why target_item_id is recommended.
    Importance = base_score(target) - score_after_mask(target)  (positive => supportive)
    If k_masked > 1, this does a simple greedy masking: remove the most impactful item,
    recompute, then remove the next most impactful, etc. (fast approximation).
    """
    # Build a list of user's history items
    if isinstance(user_vec, torch.Tensor):
        hist_items = torch.nonzero(user_vec > 0, as_tuple=False).flatten().tolist()
    else:
        hist_items = np.nonzero(user_vec > 0)[0].tolist()

    if len(hist_items) == 0:
        return {}  # no history -> nothing to explain

    # Helper to score target with current user vector
    def score_target(uvec):
        from src.utils import cosine_scores_from_user
        scores = cosine_scores_from_user(uvec, item_emb_matrix, candidate_mask=None)
        # NOTE: we allow candidate_mask=None because we want the raw score for target,
        # even if it's in history (some datasets allow this)
        if isinstance(scores, torch.Tensor):
            return float(scores[target_item_id].detach().cpu().item())
        else:
            return float(scores[target_item_id])

    # Base score with full history
    base_score = score_target(user_vec)

    # Single-mask case: report per-item importance
    if k_masked == 1:
        importances = {}
        for h in hist_items:
            # mask a single item
            if isinstance(user_vec, torch.Tensor):
                u2 = user_vec.clone()
                u2[h] = 0
            else:
                u2 = user_vec.copy()
                u2[h] = 0
            new_score = score_target(u2)
            importances[h] = base_score - new_score  # positive => removing lowers score
        return importances

    # k>1 greedy: iteratively remove the most impactful item each round
    remaining = hist_items.copy()
    removed = []
    contribs = {}  # cumulative contribution per history item
    u_curr = user_vec.clone() if isinstance(user_vec, torch.Tensor) else user_vec.copy()
    for step in range(min(k_masked, len(remaining))):
        # compute marginal impact of removing each remaining item
        deltas = {}
        curr_score = score_target(u_curr)
        for h in remaining:
            if isinstance(u_curr, torch.Tensor):
                u2 = u_curr.clone(); u2[h] = 0
            else:
                u2 = u_curr.copy(); u2[h] = 0
            s2 = score_target(u2)
            deltas[h] = curr_score - s2

        # pick the most impactful
        h_star = max(deltas, key=deltas.get)
        removed.append(h_star)
        contribs[h_star] = contribs.get(h_star, 0.0) + deltas[h_star]

        # update user vector and remaining pool
        if isinstance(u_curr, torch.Tensor):
            u_curr[h_star] = 0
        else:
            u_curr[h_star] = 0
        remaining.remove(h_star)

    # Return per-item contributions (only removed ones have nonzero values)
    return contribs


# =========================
# LIME / LIRE
# =========================
def find_lime_mask(x, item_id, recommender, item_tensor, **kw_dict):
    """
    LIME over perturbations around the user history (minus target item).
    Robust to empty histories and odd shapes from neighborhood generation.
    Returns list[(idx, weight)] filtered to items present in the user's history.
    """
    user_hist = _clone_like(x)

    # remove the target from history
    if isinstance(user_hist, torch.Tensor):
        user_hist[item_id] = 0.0
        hist_nonzero = int((user_hist > 0).sum().item())
    else:
        user_hist[item_id] = 0
        hist_nonzero = int(np.sum(np.asarray(user_hist) > 0))

    # Guard: if no features left, nothing to explain
    if hist_nonzero == 0:
        return []

    lime = LimeBase(distance_to_proximity)

    # Build neighborhood
    neighborhood_data, neighborhood_labels, distances, item_id = get_lime_args(
        user_hist, item_id, recommender, item_tensor, **kw_dict
    )

    # ---- Shape sanitization (LIME expects 2D X, 1D y, 1D dists) ----
    neighborhood_data = np.asarray(neighborhood_data, dtype=np.float32)
    if neighborhood_data.ndim == 1:
        neighborhood_data = neighborhood_data.reshape(1, -1)

    neighborhood_labels = np.asarray(neighborhood_labels, dtype=np.float32).reshape(-1)
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)

    # If shapes are still inconsistent, bail out cleanly
    if neighborhood_data.shape[0] == 0 or neighborhood_data.shape[1] == 0:
        return []
    if neighborhood_data.shape[0] != neighborhood_labels.shape[0] or neighborhood_data.shape[0] != distances.shape[0]:
        return []

    # ---- Run LIME ----
    try:
        explanation_unfiltered = lime.explain_instance_with_data(
            neighborhood_data, neighborhood_labels, distances, item_id,
            200, feature_selection="highest_weights", pos_neg="POS"
        )
    except Exception:
        return []

    # Keep only features that remain in (post-removed) history
    if isinstance(user_hist, torch.Tensor):
        hist_mask = (user_hist > 0).detach().cpu().numpy().astype(bool)
    else:
        hist_mask = (np.asarray(user_hist) > 0)

    filtered = [pair for pair in explanation_unfiltered if hist_mask[int(pair[0])]]
    return filtered




def find_lire_mask(x, item_id, recommender, **kw_dict):
    """
    Optional LIRE variant used in some code paths.
    """
    user_hist = _clone_like(x)
    if isinstance(user_hist, torch.Tensor):
        user_hist[item_id] = 0.0
    else:
        user_hist[item_id] = 0

    lime = LimeBase(distance_to_proximity)
    neighborhood_data, neighborhood_labels, distances, item_id = get_lire_args(
        user_hist, item_id, recommender, **kw_dict
    )
    most_pop_items = lime.explain_instance_with_data(
        neighborhood_data, neighborhood_labels, distances, item_id,
        200, feature_selection="highest_weights", pos_neg="POS"
    )
    return most_pop_items


# =========================
# LXR dense mask
# =========================
def find_lxr_mask(x, item_tensor, explainer, **kw):
    """
    Use your trained LXR explainer to produce a dense per-item importance vector,
    then convert to a sparse dict {i: score} for items present in the user’s history.
    """
    user_hist = x
    if isinstance(user_hist, np.ndarray):
        user_hist = torch.FloatTensor(user_hist)
    if isinstance(item_tensor, np.ndarray):
        item_tensor = torch.FloatTensor(item_tensor)
    if isinstance(item_tensor, int):
        # convert to one-hot
        n = user_hist.shape[0] if hasattr(user_hist, "shape") else len(user_hist)
        one_hot = torch.zeros(n)
        one_hot[item_tensor] = 1.0
        item_tensor = one_hot

    expl_scores = explainer(user_hist, item_tensor)
    user_hist = user_hist.to(expl_scores.device)

    # weight history by explainer scores
    x_masked = user_hist * expl_scores
    out = {}
    for i, flag in enumerate((x_masked != 0).tolist()):
        if flag:
            out[i] = float(x_masked[i].item())
    return out


# =========================
# FIA / ACCENT / SHAP
# =========================
def find_fia_mask(user_tensor, item_tensor, item_id, recommender, **kw_dict):
    """
    Feature Influence via Leave-one-out:
      infl(i) = score(user) - score(user without {i})
    Returns dict {i: infl_score} for seen items.
    """
    # make a local copy of kw without output_type to avoid duplicate kw passing
    kw_no_ot = dict(kw_dict)
    kw_no_ot.pop("output_type", None)
    device = kw_no_ot["device"]

    # Ensure single item vector is 2D: (1, num_items) for matmul in your MLP
    if isinstance(item_tensor, torch.Tensor) and item_tensor.dim() == 1:
        item_tensor = item_tensor.unsqueeze(0)

    y_pred = recommender_run(
        user_tensor, recommender, item_tensor, item_id, output_type="single", **kw_no_ot
    ).to(device)

    items_fia = {}
    user_hist = user_tensor.detach().cpu().numpy().astype(int)
    num_items = int(kw_no_ot["num_items"])

    for i in range(num_items):
        if user_hist[i] == 1:
            user_hist[i] = 0
            temp_user = torch.from_numpy(user_hist.astype(np.float32)).to(device)
            y_pred_wo = recommender_run(
                temp_user, recommender, item_tensor, item_id, output_type="single", **kw_no_ot
            ).to(device)
            infl = y_pred - y_pred_wo
            items_fia[i] = float(infl.item())
            user_hist[i] = 1
    return items_fia


def find_shapley_mask(user_tensor, user_id, model, shap_values, item_to_cluster, **kw):
    """
    If you have precomputed Shapley values:
      shap_values : array with first col = user_id, remaining = cluster contributions
      item_to_cluster : mapping item_idx -> cluster_idx
    Returns dict {i: shap_value_for_i}
    """
    item_shap = {}
    sv_user = shap_values[shap_values[:, 0].astype(int) == int(user_id)][:, 1:]
    uv = user_tensor.detach().cpu().numpy().astype(int)

    for i in np.where(uv == 1)[0]:
        cl = int(item_to_cluster[i])
        item_shap[i] = float(sv_user.T[cl][0])
    return item_shap


def find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, top_k, **kw_dict):
    """
    ACCENT: combine FIA across top-k recommended items with a telescoping sum:
      items_accent = (top_k-1) * FIA(top1) - FIA(top2) - ... - FIA(top_k)
    Returns dict {i: score}.
    """
    items_accent = defaultdict(float)
    factor = int(top_k) - 1
    device = kw_dict["device"]
    items_array = kw_dict["items_array"]

    # Rank by current user
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender_model, **kw_dict).keys())
    top_k_indices = sorted_indices[:top_k] if top_k > 1 else [sorted_indices[0]]

    # We'll remove each top-k item from the user and compute FIA for that item
    user_np = user_tensor.detach().cpu().numpy().astype(int)
    for it, item_k in enumerate(top_k_indices):
        user_np[item_k] = 0
        temp_user = torch.from_numpy(user_np.astype(np.float32)).to(device)

        # 2D item vector for safe matmul downstream
        item_vec = items_array[item_k]
        temp_item = torch.from_numpy(item_vec.astype(np.float32)).unsqueeze(0).to(device)

        fia = find_fia_mask(temp_user, temp_item, item_k, recommender_model, **kw_dict)

        if it == 0:
            for k, v in fia.items():
                items_accent[k] = v * factor
        else:
            for k, v in fia.items():
                items_accent[k] -= v

        # restore the removed top-k interaction
        user_np[item_k] = 1

    # final sign flip per ACCENT definition in the original repo
    for k in list(items_accent.keys()):
        items_accent[k] *= -1.0

    return items_accent
