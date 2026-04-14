"""
api.py  —  FastAPI backend for the News Recommender User Study
Run with:  uvicorn api:app --reload --port 8000
"""

import sys, pickle, json, csv, torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent))
from src.nu_data import load_nu_dataset
from src.explainers import find_cosine_mask

EMB_CSV   = "data/raw/mind_embeddings_10k.csv"
VIEWS_CSV = "data/raw/mind_views_10k.csv"
NEWS_TSV  = "/Users/Fehmican/Downloads/MINDsmall_train/news.tsv"
LOG_FILE  = "logs/study_interactions.csv"
TOP_N     = 30

print("Loading pipeline data...")
device = torch.device("cpu")
(_, _, _, _, train_array, _, _, _,
 _, item_embs, id2idx, idx2id) = load_nu_dataset(
    EMB_CSV, VIEWS_CSV, device,
    min_profile=5, min_test=5, split_ratio=0.8
)

news_df = pd.read_csv(
    NEWS_TSV, sep="\t", header=None,
    names=["news_id","category","subcategory","title","abstract","url","te","ae"],
    usecols=["news_id","category","subcategory","title","abstract"]
)
news_df["abstract"] = news_df["abstract"].fillna("")
id2meta = {
    row.news_id: {
        "title":       row.title,
        "category":    row.category,
        "subcategory": row.subcategory,
        "abstract":    row.abstract[:200]
    }
    for row in news_df.itertuples()
}

np.random.seed(42)
valid_users = np.where(train_array.sum(axis=1) >= 8)[0]
STUDY_USERS = np.random.choice(valid_users, size=min(200, len(valid_users)), replace=False).tolist()

def article_meta(item_idx):
    aid = idx2id.get(item_idx, str(item_idx))
    meta = id2meta.get(aid, {})
    return {
        "id":          aid,
        "item_idx":    item_idx,
        "title":       meta.get("title", aid),
        "category":    meta.get("category", "news"),
        "subcategory": meta.get("subcategory", ""),
        "abstract":    meta.get("abstract", ""),
    }

def recommend(history_indices, n=TOP_N):
    if len(history_indices) == 0:
        return []
    hist_embs = item_embs[history_indices]
    user_emb  = hist_embs.mean(dim=0, keepdim=True)
    scores    = (item_embs @ user_emb.T).squeeze().numpy()
    scores[history_indices] = -999
    top_idxs  = np.argsort(-scores)[:n]
    return [article_meta(int(i)) for i in top_idxs]

def counterfactual_for(history_indices, target_idx):
    if len(history_indices) == 0:
        return []
    CF_N = 5  # check top-5 for counterfactuals — easier to find meaningful ones
    original_scores = (item_embs @ item_embs[history_indices].mean(dim=0)).numpy()
    original_scores[history_indices] = -999
    original_top = set(int(i) for i in np.argsort(-original_scores)[:CF_N])
    results = []
    for remove_idx in history_indices:
        new_hist = np.array([x for x in history_indices if x != remove_idx])
        if len(new_hist) == 0:
            continue
        new_scores = (item_embs @ item_embs[new_hist].mean(dim=0)).numpy()
        new_scores[history_indices] = -999
        new_top = set(int(i) for i in np.argsort(-new_scores)[:CF_N])
        if target_idx in (original_top - new_top):
            results.append({"article": article_meta(int(remove_idx)), "impact": "high"})
    return results

def ensure_log():
    Path("logs").mkdir(exist_ok=True)
    if not Path(LOG_FILE).exists():
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","session_id","condition","event","data"])

def log_event(session_id, condition, event, data):
    ensure_log()
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), session_id, condition, event, json.dumps(data)])

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="News Recommender Study API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SessionRequest(BaseModel):
    condition: str
    user_idx:  Optional[int] = None

class RemoveRequest(BaseModel):
    session_id:  str
    condition:   str
    user_idx:    int
    removed_ids: List[str]

class LogRequest(BaseModel):
    session_id: str
    condition:  str
    event:      str
    data:       dict = {}

@app.get("/study")
def serve_study(request: Request):
    return FileResponse("study.html", headers={"ngrok-skip-browser-warning": "true"})

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/session/start")
def start_session(req: SessionRequest):
    import uuid
    session_id = str(uuid.uuid4())[:8]
    user_idx = req.user_idx if req.user_idx is not None else int(np.random.choice(STUDY_USERS))
    user_vec = train_array[user_idx]
    history_indices = np.where(user_vec > 0)[0]
    history = [article_meta(int(i)) for i in history_indices]
    recs    = recommend(history_indices)
    log_event(session_id, req.condition, "session_start", {"user_idx": user_idx})
    return {"session_id": session_id, "user_idx": user_idx, "condition": req.condition,
            "history": history, "recommendations": recs}

@app.post("/recommend")
def get_recommendations(req: RemoveRequest):
    user_vec = train_array[req.user_idx]
    history_indices = np.where(user_vec > 0)[0]
    removed_idxs = [id2idx[aid] for aid in req.removed_ids if aid in id2idx]
    active_history = np.array([i for i in history_indices if i not in removed_idxs])
    recs = recommend(active_history)
    explanations = {}
    if req.condition in ("B", "D"):
        for rec in recs[:5]:
            explanations[rec["id"]] = counterfactual_for(active_history, rec["item_idx"])
    log_event(req.session_id, req.condition, "recommendations_fetched",
              {"user_idx": req.user_idx, "removed_count": len(removed_idxs)})
    return {"recommendations": recs, "explanations": explanations, "removed_count": len(removed_idxs)}

@app.post("/log")
def log_interaction(req: LogRequest):
    log_event(req.session_id, req.condition, req.event, req.data)
    return {"status": "logged"}

@app.get("/study/users")
def get_study_users():
    return {"users": STUDY_USERS[:50]}
