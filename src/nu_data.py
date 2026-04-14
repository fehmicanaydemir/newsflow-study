# src/nu_data.py
import re
import numpy as np
import pandas as pd
import torch

# -----------------------------
# Embedding parsing (robust)
# -----------------------------
def _parse_embedding(x):
    """
    Parse an embedding stored as a string or list into a float32 numpy array.
    Handles both comma-separated and space-separated formats, with or without brackets.
    Examples accepted:
      "[0.1, 0.2, 0.3]"
      "[0.1  0.2  0.3]"
      "0.1,0.2,0.3"
      "0.1 0.2 0.3"
    """
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=np.float32)

    if pd.isna(x):
        return None

    s = str(x).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()

    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'\s+', ' ', s)

    if ',' in s:
        arr = np.fromstring(s, sep=',', dtype=np.float32)
    else:
        arr = np.fromstring(s, sep=' ', dtype=np.float32)

    return arr


def load_item_embeddings(emb_csv_path, device):
    df = pd.read_csv(emb_csv_path)

    # expected cols: SOURCE_SHORT_ARTICLE_ID, embedding
    df = df.dropna(subset=["SOURCE_SHORT_ARTICLE_ID", "embedding"]).copy()

    # Parse embeddings
    df["embedding"] = df["embedding"].apply(_parse_embedding)

    # Keep only valid vectors
    mask_ok_type = df["embedding"].apply(lambda v: isinstance(v, np.ndarray) and v.size > 0)
    df = df[mask_ok_type].reset_index(drop=True)

    # Ensure consistent dimension
    dims = df["embedding"].apply(lambda v: v.shape[0])
    most_common_dim = dims.mode().iloc[0]
    df = df[dims == most_common_dim].reset_index(drop=True)

    # Drop rows with NaNs/Infs
    mask_ok = df["embedding"].apply(lambda v: np.isfinite(v).all())
    df = df[mask_ok].reset_index(drop=True)

    ids = df["SOURCE_SHORT_ARTICLE_ID"].astype(str).tolist()
    id2idx = {aid: i for i, aid in enumerate(ids)}
    idx2id = {i: aid for aid, i in id2idx.items()}

    mat = np.stack(df["embedding"].to_list(), axis=0).astype(np.float32)  # [I, D]
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    mat = mat / norms
    item_emb_matrix = torch.from_numpy(mat).to(device).float()
    return id2idx, idx2id, item_emb_matrix


# -----------------------------
# Per-user split policy
# -----------------------------
def _split_user_history_by_time(items_sorted_idx, split_ratio=0.8, max_profile=100):
    """
    Policy:
      - Cap each user's history to the last `max_profile` reads (by time order).
      - Always do an 80/20 split by time (train = earliest part, test = latest part).
      - Ensure at least 1 test item (so small users still have a test).
      - Ensure at least 1 train item (skip users with <2 items after capping).

    items_sorted_idx: 1D array/list of item indices sorted by timestamp ASC.
    Returns (train_items, test_items) or (None, None) if too small to split.
    """
    if max_profile is not None and max_profile > 0:
        # keep the most recent max_profile items
        items_sorted_idx = items_sorted_idx[-max_profile:]

    n = len(items_sorted_idx)
    if n < 2:
        return None, None  # need at least 1 train + 1 test

    test_k = max(1, int(round((1.0 - float(split_ratio)) * n)))  # at least 1 test item
    train_k = n - test_k
    if train_k < 1:
        # force at least 1 train item
        train_k = 1
        test_k = n - 1

    train_items = items_sorted_idx[:train_k]
    test_items  = items_sorted_idx[train_k:]
    return train_items, test_items


def build_user_item_arrays(views_csv_path, id2idx,
                           min_profile=5, min_test=5, split_ratio=0.8,
                           max_profile=100):
    """
    NOTE:
      - `min_test` is ignored (kept only for signature compatibility).
      - We always do an 80/20 split with at least 1 test item.
      - Users with total reads < `min_profile` are skipped (optional).
      - Each user's history is capped to the last `max_profile` reads.
    """
    df = pd.read_csv(views_csv_path)
    # expected cols: TIMESTAMP, USER_ID, ARTICLE_ID, ...
    df = df.dropna(subset=["TIMESTAMP", "USER_ID", "ARTICLE_ID"]).copy()
    df["ARTICLE_ID"] = df["ARTICLE_ID"].astype(str)

    # Align to items that have embeddings
    df = df[df["ARTICLE_ID"].isin(id2idx)]
    if df.empty:
        raise ValueError("No view events left after aligning with embeddings.")

    df["item_idx"] = df["ARTICLE_ID"].map(id2idx)
    df["ts"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df.dropna(subset=["ts"])

    num_items = len(id2idx)
    train_rows, test_rows = [], []
    kept_users = []

    # Pre-sort by time once, then group
    df = df.sort_values(["USER_ID", "ts"])

    for uid, g in df.groupby("USER_ID", sort=False):
        items = g["item_idx"].to_numpy(dtype=np.int32)

        # Optional minimum profile filter (kept for your CLI compatibility)
        if min_profile and len(items) < int(min_profile):
            continue

        # Split by time with cap + 80/20
        train_items, test_items = _split_user_history_by_time(
            items_sorted_idx=items,
            split_ratio=split_ratio,
            max_profile=max_profile
        )
        if train_items is None:
            continue

        # Binary incidence vectors over unique items
        train_row = np.zeros(num_items, dtype=np.float32)
        test_row  = np.zeros(num_items, dtype=np.float32)
        if len(train_items) > 0:
            train_row[np.unique(train_items)] = 1.0
        if len(test_items) > 0:
            test_row[np.unique(test_items)] = 1.0

        train_rows.append(train_row)
        test_rows.append(test_row)
        kept_users.append(uid)

    if not train_rows:
        raise ValueError("No users passed the NU split policy (check min_profile, data alignment, etc.).")

    train_array = np.stack(train_rows, axis=0)
    test_array  = np.stack(test_rows, axis=0)

    # Items array: one-hot identity
    items_array = np.eye(num_items, dtype=np.float32)

    # Popularity over ALL remaining interactions (after alignment)
    pop = df["item_idx"].value_counts().to_dict()
    pop_array = np.zeros(num_items, dtype=np.float32)
    for i, c in pop.items():
        if 0 <= i < num_items:
            pop_array[i] = float(c)

    return kept_users, train_array, test_array, items_array, pop_array


def load_nu_dataset(emb_csv_path, views_csv_path, device,
                    min_profile=5, min_test=5, split_ratio=0.8, max_profile=100):
    """
    Public loader used by your scripts. Keeps the same signature as before,
    but internally enforces:
      - capping to last `max_profile` reads,
      - 80/20 split with >=1 test item,
      - `min_test` is ignored.
    """
    id2idx, idx2id, item_embs = load_item_embeddings(emb_csv_path, device)

    users, train_array, test_array, items_array, pop_array = build_user_item_arrays(
        views_csv_path, id2idx,
        min_profile=min_profile,
        min_test=min_test,          # ignored inside
        split_ratio=split_ratio,    # typically 0.8
        max_profile=max_profile     # cap at 100 by default
    )

    all_items_tensor = torch.eye(len(id2idx), dtype=torch.float32, device=device)

    # match the tuple structure used by load_and_preprocess_data
    train_data = None
    test_data = None
    static_test_data = pd.DataFrame(
        np.concatenate(
            [test_array, np.zeros((test_array.shape[0], 2), dtype=np.float32)],
            axis=1
        )
    )

    return (
        train_data, test_data, static_test_data, pop_array,
        train_array, test_array, items_array, all_items_tensor,
        pop_array, item_embs, id2idx, idx2id
    )
