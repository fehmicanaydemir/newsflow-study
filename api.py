"""
api.py  —  FastAPI backend for the News Recommender User Study
Run with:  uvicorn api:app --reload --port 8000
"""

import sys, pickle, json, csv, torch
import anthropic
from scipy.sparse import csr_matrix, load_npz, save_npz
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

EMB_CSV   = "data/raw/mind_embeddings_full_384.csv"
VIEWS_CSV = "data/raw/mind_views.csv"
NEWS_TSV  = "data/raw/news.tsv"
LOG_FILE  = "logs/study_interactions.csv"
TOP_N     = 100

# Set of valid item indices -- populated after id2meta is built
VALID_INDICES = None

import hashlib, os

CACHE_FILE = "data/processed/api_cache.pkl"

def _cache_key():
    h = hashlib.md5()
    for f in [EMB_CSV, VIEWS_CSV]:
        if os.path.exists(f):
            h.update(str(os.path.getmtime(f)).encode())
            h.update(str(os.path.getsize(f)).encode())
    return h.hexdigest()

def load_with_cache():
    key = _cache_key()
    if os.path.exists(CACHE_FILE):
        print("Checking cache...", flush=True)
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == key:
            print("Cache hit -- loading from disk (fast startup)", flush=True)
            return cached["train_array"], cached["item_embs"], cached["id2idx"], cached["idx2id"]
        print("Cache stale -- reloading", flush=True)
    else:
        print("No cache -- building from CSV (one-time, ~2 min)", flush=True)

    # Fast direct loader -- bypasses train/test split logic
    print("  Step 1/3: Reading embeddings CSV...", flush=True)
    import ast
    emb_df = pd.read_csv(EMB_CSV)
    emb_df.columns = [c.strip() for c in emb_df.columns]
    id_col  = "SOURCE_SHORT_ARTICLE_ID"
    emb_col = "embedding"

    print("  Step 2/3: Parsing embeddings...", flush=True)
    id2idx = {}
    idx2id = {}
    emb_list = []
    for i, row in enumerate(emb_df.itertuples(index=False)):
        aid = str(getattr(row, id_col))
        emb = np.array(ast.literal_eval(str(getattr(row, emb_col))), dtype=np.float32)
        id2idx[aid] = i
        idx2id[i]   = aid
        emb_list.append(emb)
    item_embs = torch.tensor(np.stack(emb_list), dtype=torch.float32)
    n_items   = len(id2idx)
    print(f"  Loaded {n_items} article embeddings, dim={item_embs.shape[1]}", flush=True)

    print("  Step 3/3: Building interaction matrix...", flush=True)
    views_df = pd.read_csv(VIEWS_CSV)
    uid_col  = "USER_ID"   if "USER_ID"   in views_df.columns else views_df.columns[1]
    aid_col  = "ARTICLE_ID" if "ARTICLE_ID" in views_df.columns else views_df.columns[2]
    users    = views_df[uid_col].astype(str).unique()
    u2idx    = {u: i for i, u in enumerate(users)}
    n_users  = len(u2idx)
    rows, cols = [], []
    for row in views_df.itertuples(index=False):
        uid = str(getattr(row, uid_col))
        aid = str(getattr(row, aid_col))
        if uid in u2idx and aid in id2idx:
            rows.append(u2idx[uid]); cols.append(id2idx[aid])
    print(f"  Interaction matrix: {n_users} users x {n_items} items", flush=True)

    print("Saving cache...", flush=True)
    os.makedirs("data/processed", exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        data = np.ones(len(rows), dtype=np.float32)
        train_array = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
        pickle.dump({"key": key, "train_array": train_array, "item_embs": item_embs,
                     "id2idx": id2idx, "idx2id": idx2id}, f, protocol=4)
    print("Cache saved. Future startups will be fast.", flush=True)
    return train_array, item_embs, id2idx, idx2id

print("Loading pipeline data...", flush=True)
device = torch.device("cpu")
train_array, item_embs, id2idx, idx2id = load_with_cache()

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
    if row.title and str(row.title).strip()
    and row.abstract and str(row.abstract).strip()
    and not str(row.title).strip().startswith("N")
    and len(str(row.title).strip()) > 10
}

# Build set of valid item indices -- only articles with proper metadata
VALID_INDICES = np.array([
    id2idx[aid] for aid in id2meta.keys() if aid in id2idx
], dtype=np.int64)
print(f"Valid articles with metadata: {len(VALID_INDICES)} / {len(id2idx)}")

np.random.seed(42)
valid_users = np.where(np.asarray(train_array.sum(axis=1)).flatten() >= 8)[0]
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
    mask = np.full(len(scores), -999.0)
    if VALID_INDICES is not None and len(VALID_INDICES) > 0:
        mask[VALID_INDICES] = scores[VALID_INDICES]
    else:
        mask = scores.copy()
    mask[history_indices] = -999.0
    top_idxs = np.argsort(-mask)[:n]
    results = []
    for i in top_idxs:
        aid = idx2id.get(int(i), "")
        if aid in id2meta:
            results.append(article_meta(int(i)))
        if len(results) >= n:
            break
    return results

def recommend_from_vec(user_vec, history_indices, n=TOP_N):
    # Build weighted embedding from user vector
    nonzero = np.where(user_vec > 0)[0]
    if len(nonzero) == 0:
        return []
    weights = torch.tensor(user_vec[nonzero], dtype=torch.float32).unsqueeze(1)
    hist_embs = item_embs[nonzero]
    user_emb  = (hist_embs * weights).sum(dim=0, keepdim=True)
    norm = user_emb.norm() + 1e-8
    user_emb = user_emb / norm
    scores = (item_embs @ user_emb.T).squeeze().numpy()
    # Mask out invalid articles (no metadata) and already-seen articles
    mask = np.full(len(scores), -999.0)
    if VALID_INDICES is not None and len(VALID_INDICES) > 0:
        mask[VALID_INDICES] = scores[VALID_INDICES]
    else:
        mask = scores.copy()
    mask[history_indices] = -999.0
    top_idxs = np.argsort(-mask)[:n]
    # Final safety filter -- only return articles that have metadata
    results = []
    for i in top_idxs:
        aid = idx2id.get(int(i), "")
        if aid in id2meta:
            results.append(article_meta(int(i)))
        if len(results) >= n:
            break
    return results

def counterfactual_for(history_indices, target_idx):
    if len(history_indices) == 0:
        return []
    CF_N_SEARCH  = 50  # search top-50 to maximise counterfactual coverage
    CF_N_DISPLAY = 3   # show top 3 counterfactuals

    original_scores = (item_embs @ item_embs[history_indices].mean(dim=0)).numpy()
    original_top = set(int(i) for i in np.argsort(-original_scores)[:CF_N_SEARCH])

    hard_candidates = []
    soft_candidates = []

    for remove_idx in history_indices:
        new_hist = np.array([x for x in history_indices if x != remove_idx])
        if len(new_hist) == 0:
            continue
        new_scores = (item_embs @ item_embs[new_hist].mean(dim=0)).numpy()
        new_top = set(int(i) for i in np.argsort(-new_scores)[:CF_N_SEARCH])
        score_drop = float(original_scores[target_idx] - new_scores[target_idx])

        if target_idx in (original_top - new_top):
            hard_candidates.append({
                "article":    article_meta(int(remove_idx)),
                "impact":     "high",
                "score_drop": score_drop
            })
        else:
            soft_candidates.append({
                "article":    article_meta(int(remove_idx)),
                "impact":     "high",
                "score_drop": score_drop
            })

    # Prefer genuine counterfactuals; fill remaining slots with largest score drops
    hard_candidates.sort(key=lambda x: x["score_drop"], reverse=True)
    soft_candidates.sort(key=lambda x: x["score_drop"], reverse=True)

    combined = hard_candidates[:CF_N_DISPLAY]
    if len(combined) < CF_N_DISPLAY:
        remaining = CF_N_DISPLAY - len(combined)
        combined += soft_candidates[:remaining]

    results = combined
    for r in results:
        r.pop("score_drop", None)

    # Topic-based explanation -- always add
    target_emb = item_embs[target_idx]
    hist_embs  = item_embs[history_indices]
    sims       = (hist_embs @ target_emb).numpy()
    top5_hist  = np.argsort(-sims)[:5]
    top_cats   = []
    seen_cats  = set()
    for idx in top5_hist:
        meta = article_meta(int(history_indices[idx]))
        cat  = meta.get("category", "").strip().lower()
        if cat and cat not in seen_cats:
            top_cats.append(cat.capitalize())
            seen_cats.add(cat)
    if top_cats:
        results.append({
            "impact": "topic",
            "topics": top_cats[:3]
        })

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
from fastapi.responses import FileResponse, HTMLResponse

# In-memory session store — maps session_id -> synthetic user_vec
SESSION_STORE = {}

app = FastAPI(title="News Recommender Study API")

# Disable access logging to avoid storing IP addresses (GDPR)
import logging
logging.getLogger("uvicorn.access").disabled = True
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SessionRequest(BaseModel):
    condition:   str
    user_idx:    Optional[int] = None
    topics:      Optional[list] = None
    warmup_ids:  Optional[list] = None

class RemoveRequest(BaseModel):
    session_id:   str
    condition:    str
    user_idx:     int
    removed_ids:  List[str]
    selected_ids: Optional[List[str]] = None
    read_ids:     Optional[List[str]] = None

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
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/study")


# ── Admin Stats ──────────────────────────────────────────────────────────────

@app.get("/admin/stats", response_class=HTMLResponse)
def admin_stats():
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        return HTMLResponse("<h2>No log file found yet.</h2>")
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return HTMLResponse("<h2>Log file is empty — no participants yet.</h2>")
    from collections import defaultdict, Counter
    cond_events = defaultdict(Counter)
    sessions_by_cond = defaultdict(set)
    surveys_by_cond  = defaultdict(set)
    recent = []
    for row in rows:
        cond  = row.get("condition", "?")
        event = row.get("event", "?")
        sid   = row.get("session_id", "?")
        ts    = row.get("timestamp", "")
        cond_events[cond][event] += 1
        if event == "session_start":
            sessions_by_cond[cond].add(sid)
        if event == "post_survey":
            surveys_by_cond[cond].add(sid)
        recent.append({"ts": ts, "sid": sid, "cond": cond, "event": event})
    recent.sort(key=lambda x: x["ts"], reverse=True)
    recent = recent[:20]
    all_conds = ["A","B","C","D"]
    total_sessions = sum(len(v) for v in sessions_by_cond.values())
    total_surveys  = sum(len(v) for v in surveys_by_cond.values())
    def row_color(cond):
        return {"A":"#fff8f0","B":"#f0f8ff","C":"#f0fff4","D":"#fff0f8"}.get(cond,"#fff")
    cond_rows = ""
    for c in all_conds:
        n_sess   = len(sessions_by_cond[c])
        n_survey = len(surveys_by_cond[c])
        pct      = f"{n_sess/max(total_sessions,1)*100:.0f}%"
        bar_w    = f"{n_sess/max(total_sessions,1)*200:.0f}px"
        label    = {"A":"No exp · No ctrl","B":"Exp · No ctrl","C":"No exp · Ctrl","D":"Exp · Ctrl"}[c]
        cond_rows += f'''<tr style="background:{row_color(c)}"><td><strong>Condition {c}</strong><br><small style="color:#888">{label}</small></td><td style="text-align:center">{n_sess}</td><td style="text-align:center">{n_survey}</td><td style="text-align:center">{pct}</td><td><div style="background:#0f0f0f;height:14px;width:{bar_w};border-radius:2px;min-width:2px"></div></td></tr>'''
    all_events = sorted(set(e for ec in cond_events.values() for e in ec.keys()))
    event_header = "".join(f"<th>{c}</th>" for c in all_conds)
    event_rows = ""
    for ev in all_events:
        cells = "".join(f"<td style='text-align:center'>{cond_events[c].get(ev,0)}</td>" for c in all_conds)
        event_rows += f"<tr><td><code>{ev}</code></td>{cells}</tr>"
    recent_rows = ""
    for r in recent:
        recent_rows += f'''<tr style="background:{row_color(r['cond'])}"><td style="font-size:11px;color:#888">{r['ts'][:19]}</td><td><code>{r['sid']}</code></td><td style="text-align:center"><strong>{r['cond']}</strong></td><td><code>{r['event']}</code></td></tr>'''
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/><meta http-equiv="refresh" content="30"/><title>NewsFlow — Study Stats</title><style>body{{font-family:'Helvetica Neue',sans-serif;background:#f5f0e8;color:#0f0f0f;padding:2rem;}}h1{{font-size:1.6rem;margin-bottom:0.2rem;}}h2{{font-size:1rem;font-weight:600;margin:1.8rem 0 0.6rem;text-transform:uppercase;letter-spacing:.1em;color:#888;}}.meta{{font-size:0.8rem;color:#888;margin-bottom:2rem;}}.cards{{display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap;}}.card{{background:#fff;border:1px solid #c8bfa8;border-radius:4px;padding:1rem 1.5rem;min-width:140px;}}.card .num{{font-size:2rem;font-weight:700;}}.card .lbl{{font-size:0.75rem;color:#888;text-transform:uppercase;letter-spacing:.08em;}}table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #c8bfa8;border-radius:4px;overflow:hidden;margin-bottom:2rem;}}th{{background:#0f0f0f;color:#fff;padding:8px 12px;font-size:0.78rem;letter-spacing:.06em;text-transform:uppercase;text-align:left;}}td{{padding:8px 12px;border-bottom:1px solid #ede8dc;font-size:0.85rem;}}tr:last-child td{{border-bottom:none;}}</style></head><body><h1>NewsFlow — Study Dashboard</h1><div class="meta">Auto-refreshes every 30s · {datetime.now().strftime("%d %b %Y %H:%M:%S")} · {len(rows)} total log entries</div><div class="cards"><div class="card"><div class="num">{total_sessions}</div><div class="lbl">Sessions started</div></div><div class="card"><div class="num">{total_surveys}</div><div class="lbl">Surveys completed</div></div><div class="card"><div class="num">{total_surveys}/{max(total_sessions,1)}</div><div class="lbl">Completion rate</div></div></div><h2>Condition Distribution</h2><table><thead><tr><th>Condition</th><th>Sessions</th><th>Completed</th><th>Share</th><th>Bar</th></tr></thead><tbody>{cond_rows}</tbody></table><h2>Events by Condition</h2><table><thead><tr><th>Event</th>{event_header}</tr></thead><tbody>{event_rows}</tbody></table><h2>Recent Activity</h2><table><thead><tr><th>Timestamp</th><th>Session</th><th>Cond</th><th>Event</th></tr></thead><tbody>{recent_rows}</tbody></table></body></html>"""
    return HTMLResponse(html)

# ── End Admin Stats ───────────────────────────────────────────────────────────
@app.post("/session/start")
def start_session(req: SessionRequest):
    import uuid
    session_id = str(uuid.uuid4())[:8]

    # Build user vector purely from warmup article selections
    warmup_indices = []
    if req.warmup_ids:
        for wid in req.warmup_ids:
            if wid in id2idx:
                warmup_indices.append(id2idx[wid])

    if len(warmup_indices) > 0:
        history_indices = np.array(warmup_indices)

        # ── Warmup signal (70%) ──────────────────────────────────────────
        warmup_vec = np.zeros(len(id2idx), dtype=np.float32)
        for idx in history_indices:
            warmup_vec[idx] = 1.0

        # ── Topic signal (30%) ───────────────────────────────────────────
        # Build a soft prior from selected topics by sampling articles per topic
        topic_vec = np.zeros(len(id2idx), dtype=np.float32)
        if req.topics and len(req.topics) > 0:
            topic_set = set(t.lower() for t in req.topics)
            SAMPLES_PER_TOPIC = 5
            topic_sample_indices = []
            for aid, meta in id2meta.items():
                cat = meta.get("category", "").lower()
                if cat in topic_set and aid in id2idx:
                    topic_sample_indices.append(id2idx[aid])
            if len(topic_sample_indices) > 0:
                np.random.seed(None)
                n_sample = min(len(req.topics) * SAMPLES_PER_TOPIC, len(topic_sample_indices))
                sampled  = np.random.choice(topic_sample_indices, size=n_sample, replace=False)
                for idx in sampled:
                    topic_vec[idx] = 1.0

        # ── Blend: 70% warmup + 30% topic ────────────────────────────────
        WARMUP_WEIGHT = 0.7
        TOPIC_WEIGHT  = 0.3
        if topic_vec.sum() > 0:
            user_vec = WARMUP_WEIGHT * warmup_vec + TOPIC_WEIGHT * topic_vec
        else:
            user_vec = warmup_vec  # fallback if no topic signal

        # Exclude warmup articles from feed (already seen)
        recs    = recommend_from_vec(user_vec, history_indices)
        history = [article_meta(int(i)) for i in history_indices]

        SESSION_STORE[session_id] = {
            "user_vec":        user_vec.copy(),
            "selected_ids":    [idx2id.get(i, str(i)) for i in warmup_indices],
            "read_ids":        [],
            "history_indices": history_indices.tolist()
        }
        log_event(session_id, req.condition, "session_start", {
            "topics":         req.topics,
            "warmup_count":   len(warmup_indices),
            "topic_blend":    True,
            "warmup_weight":  WARMUP_WEIGHT,
            "topic_weight":   TOPIC_WEIGHT
        })
        return {
            "session_id":      session_id,
            "user_idx":        0,
            "condition":       req.condition,
            "history":         history,
            "recommendations": recs
        }

    # Fallback if no warmup selections
    user_idx = int(np.random.choice(STUDY_USERS))
    user_vec = np.asarray(train_array[user_idx]).flatten()
    history_indices = np.where(user_vec > 0)[0]
    history = [article_meta(int(i)) for i in history_indices]
    recs    = recommend(history_indices)
    SESSION_STORE[session_id] = {
        "user_vec":        user_vec.copy(),
        "selected_ids":    [idx2id.get(i, str(i)) for i in history_indices],
        "read_ids":        [],
        "history_indices": history_indices.tolist()
    }
    log_event(session_id, req.condition, "session_start", {
        "topics":         req.topics,
        "warmup_count":   0,
        "pure_selection": False
    })
    return {
        "session_id":      session_id,
        "user_idx":        user_idx,
        "condition":       req.condition,
        "history":         history,
        "recommendations": recs
    }

class ExplainBatchRequest(BaseModel):
    session_id: str
    article_ids: list

@app.post("/explain/batch")
def explain_batch(req: ExplainBatchRequest):
    sid = req.session_id
    if sid not in SESSION_STORE:
        return {"explanations": {}}
    sess = SESSION_STORE[sid]
    active_arr = np.zeros(len(id2idx), dtype=np.float32)
    for aid in sess.get("active_ids", []):
        if aid in id2idx:
            active_arr[id2idx[aid]] = 1.0
    result = {}
    for article_id in req.article_ids[:20]:
        if article_id in id2idx:
            result[article_id] = counterfactual_for(active_arr, id2idx[article_id])
    return {"explanations": result}

@app.post("/recommend")
def get_recommendations(req: RemoveRequest):
    removed_set  = set(req.removed_ids or [])
    selected_ids = list(req.selected_ids or [])
    read_ids     = list(req.read_ids or [])

    # Update session store with latest read_ids
    if req.session_id in SESSION_STORE:
        stored = SESSION_STORE[req.session_id]
        existing_read = set(stored.get("read_ids", []))
        for rid in read_ids:
            if rid not in existing_read:
                stored["read_ids"].append(rid)
                existing_read.add(rid)
        if not selected_ids:
            selected_ids = stored.get("selected_ids", [])
        read_ids = stored.get("read_ids", [])

    # Active history = selected + read, minus removed
    all_source_ids  = list(dict.fromkeys(selected_ids + read_ids))
    active_ids      = [aid for aid in all_source_ids if aid not in removed_set]
    active_indices  = [id2idx[aid] for aid in active_ids if aid in id2idx]

    # Exclude all seen articles from feed (selected + read)
    all_seen_indices = np.array([id2idx[aid] for aid in all_source_ids if aid in id2idx])

    if len(active_indices) > 0:
        active_arr = np.array(active_indices)
        user_vec   = np.zeros(len(id2idx), dtype=np.float32)
        for idx in active_arr:
            user_vec[idx] = 1.0
        recs = recommend_from_vec(user_vec, all_seen_indices if len(all_seen_indices) > 0 else active_arr)
    else:
        # No active history -- return random articles from valid pool
        import random as _random
        valid_ids = [aid for aid in id2meta.keys() if aid in id2idx]
        _random.shuffle(valid_ids)
        seen_set = set(all_source_ids)
        random_recs = []
        for aid in valid_ids:
            if aid not in seen_set:
                random_recs.append(article_meta(id2idx[aid]))
            if len(random_recs) >= TOP_N:
                break
        recs = random_recs

    # Explanations fetched separately via /explain/batch
    explanations = {}

    empty_history = len(active_indices) == 0
    log_event(req.session_id, req.condition, "recommendations_fetched", {
        "selected_count": len(selected_ids),
        "read_count":     len(read_ids),
        "removed_count":  len(removed_set),
        "active_count":   len(active_indices),
        "empty_history":  empty_history
    })
    return {
        "recommendations": recs,
        "explanations":    explanations,
        "active_count":    len(active_indices),
        "empty_history":   empty_history
    }

@app.post("/log")
def log_interaction(req: LogRequest):
    log_event(req.session_id, req.condition, req.event, req.data)
    return {"status": "logged"}

@app.get("/study/users")
def get_study_users():
    return {"users": STUDY_USERS[:50]}

class WarmupRequest(BaseModel):
    topics: list
    exclude_ids: list = []

class MatchRequest(BaseModel):
    topics: list
    warmup_ids: list = []

@app.post("/match/user")
def match_user(req: MatchRequest):
    if not req.topics:
        user_idx = int(np.random.choice(STUDY_USERS))
        return {"user_idx": user_idx, "match_score": 0.0}

    topic_set = set(t.lower() for t in req.topics)
    best_user = None
    best_score = -1

    for uid in STUDY_USERS:
        user_vec = np.asarray(train_array[uid]).flatten()
        history_indices = np.where(user_vec > 0)[0]
        if len(history_indices) == 0:
            continue

        match_count = 0
        total = len(history_indices)
        for idx in history_indices:
            aid  = idx2id.get(idx, "")
            meta = id2meta.get(aid, {})
            cat  = meta.get("category", "").lower()
            if cat in topic_set:
                match_count += 1

        score = match_count / max(total, 1)

        for warmup_id in req.warmup_ids:
            if warmup_id in id2idx:
                if train_array[uid, id2idx[warmup_id]] > 0:
                    score += 0.1

        if score > best_score:
            best_score = score
            best_user  = uid

    user_idx = int(best_user) if best_user is not None else int(np.random.choice(STUDY_USERS))
    return {"user_idx": user_idx, "match_score": float(best_score)}

@app.post("/warmup/articles")
def get_warmup_articles(req: WarmupRequest):
    import random
    topic_set   = set(t.lower() for t in (req.topics or []))
    exclude_set = set(req.exclude_ids or [])
    all_ids     = [aid for aid in id2meta.keys() if aid not in exclude_set]
    random.shuffle(all_ids)

    TARGET      = 50
    TOPIC_SHARE = 0.70  # 70% from selected topics
    OTHER_SHARE = 0.30  # 30% from other categories

    n_topic = int(TARGET * TOPIC_SHARE)  # 35 from selected topics
    n_other = TARGET - n_topic           # 15 from other categories

    topic_articles = []
    other_articles = []
    seen_titles    = set()

    # Track seen subcategories to avoid clustering similar stories
    seen_subcats  = {}  # subcat -> count, cap per subcat
    MAX_PER_SUBCAT = 2  # max 2 articles per subcategory

    def title_fingerprint(title):
        # First 4 words as a rough dedup key
        words = title.lower().split()[:4]
        return " ".join(words)

    seen_fingerprints = set()

    for aid in all_ids:
        if len(topic_articles) >= n_topic and len(other_articles) >= n_other:
            break
        meta    = id2meta.get(aid, {})
        cat     = meta.get("category", "").lower()
        subcat  = meta.get("subcategory", "").lower()
        title   = meta.get("title", "")
        if not title or title in seen_titles:
            continue

        # Skip if title fingerprint already seen (catches "Three takeaways from X" clusters)
        fp = title_fingerprint(title)
        if fp in seen_fingerprints:
            continue

        # Skip if subcategory already has MAX_PER_SUBCAT articles
        if subcat and seen_subcats.get(subcat, 0) >= MAX_PER_SUBCAT:
            continue

        article = {
            "id":       aid,
            "title":    title,
            "category": meta.get("category", "news"),
        }
        if topic_set and cat in topic_set:
            if len(topic_articles) < n_topic:
                topic_articles.append(article)
                seen_titles.add(title)
                seen_fingerprints.add(fp)
                seen_subcats[subcat] = seen_subcats.get(subcat, 0) + 1
        else:
            if len(other_articles) < n_other:
                other_articles.append(article)
                seen_titles.add(title)
                seen_fingerprints.add(fp)
                seen_subcats[subcat] = seen_subcats.get(subcat, 0) + 1

    # Merge and shuffle so topic and other articles are interleaved
    combined = topic_articles + other_articles
    random.shuffle(combined)
    return {"articles": combined[:TARGET]}
class ArticleRequest(BaseModel):
    title:    str
    abstract: str
    category: str

@app.post("/generate/article")
def generate_article(req: ArticleRequest):
    prompt = f"""You are a staff writer at a major online news outlet. Your editor has given you a headline and summary — expand it into a publishable article.

Title: {req.title}
Category: {req.category}
Summary: {req.abstract}

Writing guidelines:
- Length: 150-300 words depending on the story. Breaking news is shorter. Features and analysis run longer.
- Structure: vary it. Lead with the most important fact, a scene, a statistic, or a question — whichever fits the story best.
- Paragraphs: 2 to 4. Not every story needs the same shape.
- Tone: match the category. Sports can be energetic. Finance should be measured. Health should be clear and calm. News should be direct.
- Do not repeat the title word-for-word in the opening sentence.
- Do not invent quotes, names, numbers, or details not in the summary.
- Do not include a headline, byline, date, or section label.
- Write in third person.
- End naturally — not every article needs a forward-looking conclusion. Sometimes the story just ends."""

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system="You are a staff writer at a major online news outlet. Write in plain text only. No markdown, no headers, no bold, no italics, no formatting of any kind. Just clean paragraphs.",
            messages=[{"role": "user", "content": prompt}]
        )
        article_text = response.content[0].text.strip()
        # Strip markdown headers and formatting
        import re
        article_text = re.sub(r'^#{1,6}\s+.*\n?', '', article_text, flags=re.MULTILINE)
        article_text = re.sub(r'\*\*(.*?)\*\*', r'\1', article_text)
        article_text = re.sub(r'\*(.*?)\*', r'\1', article_text)
        article_text = article_text.strip()
        return {"text": article_text, "status": "ok"}
    except Exception as e:
        return {"text": req.abstract, "status": "fallback", "error": str(e)}
