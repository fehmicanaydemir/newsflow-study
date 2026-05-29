import pandas as pd
from sentence_transformers import SentenceTransformer

OUT_CSV  = "data/raw/mind_embeddings_full_384.csv"
NEWS_TSV = "/Users/Fehmican/Downloads/MINDsmall_train/news.tsv"
BATCH_SIZE = 512

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

print("Loading model: all-MiniLM-L6-v2 (384d)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open(OUT_CSV, "w") as f:
    f.write("SOURCE_SHORT_ARTICLE_ID,embedding\n")

texts  = all_articles["text"].tolist()
ids    = all_articles["news_id"].tolist()
written = 0

with open(OUT_CSV, "a") as f:
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_ids   = ids[i:i+BATCH_SIZE]
        embeddings  = model.encode(batch_texts, batch_size=128,
                                   show_progress_bar=False, convert_to_numpy=True)
        for aid, emb in zip(batch_ids, embeddings):
            f.write(f'{aid},"{emb.tolist()}"\n')
        written += len(batch_ids)
        print(f"  {written}/{len(texts)} ({100*written//len(texts)}%)", flush=True)

print(f"Done. {written} articles written to {OUT_CSV}")
