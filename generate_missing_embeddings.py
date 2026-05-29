"""
Generate embeddings for all MIND articles using all-mpnet-base-v2 (768d)
Processes in batches to avoid memory issues.
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

OUT_CSV  = "data/raw/mind_embeddings_full.csv"
NEWS_TSV = "/Users/Fehmican/Downloads/MINDsmall_train/news.tsv"
BATCH_SIZE = 256

print("Loading news.tsv...")
news = pd.read_csv(
    NEWS_TSV, sep="\t", header=None,
    names=["news_id","category","subcategory","title","abstract","url","te","ae"],
    usecols=["news_id","title","abstract"]
)
news["abstract"] = news["abstract"].fillna("")
news["text"] = news["title"].fillna("") + " " + news["abstract"]

all_articles = news[
    news.title.notna() &
    (news.title.str.strip().str.len() > 10) &
    news.abstract.notna() &
    (news.abstract.str.strip().str.len() > 5)
].copy()
print(f"Articles to embed: {len(all_articles)}")

print("Loading model: all-mpnet-base-v2 (768 dimensions)...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Write header
with open(OUT_CSV, "w") as f:
    f.write("SOURCE_SHORT_ARTICLE_ID,embedding\n")

texts  = all_articles["text"].tolist()
ids    = all_articles["news_id"].tolist()
written = 0

print(f"Generating embeddings in batches of {BATCH_SIZE}...")
with open(OUT_CSV, "a") as f:
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_ids   = ids[i:i+BATCH_SIZE]
        embeddings  = model.encode(
            batch_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        for aid, emb in zip(batch_ids, embeddings):
            f.write(f'{aid},"{emb.tolist()}"\n')
        written += len(batch_ids)
        pct = 100 * written // len(texts)
        print(f"  {written}/{len(texts)} ({pct}%)", flush=True)

print(f"Done. Output: {OUT_CSV}")
print(f"Total articles embedded: {written}")
