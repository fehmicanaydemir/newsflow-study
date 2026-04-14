"""
prepare_mind_data.py
Converts raw MIND-small files into the two CSVs expected by the pipeline.
"""
import argparse, os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

EMB_ID_COL   = "SOURCE_SHORT_ARTICLE_ID"
EMB_VEC_COL  = "embedding"
VIEW_TIME    = "TIMESTAMP"
VIEW_USER    = "USER_ID"
VIEW_ARTICLE = "ARTICLE_ID"

NEWS_COLS     = ["news_id","category","subcategory","title","abstract","url","title_entities","abstract_entities"]
BEHAVIOR_COLS = ["impression_id","user_id","time","history","impressions"]

def load_news(path):
    df = pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS, usecols=["news_id","title","abstract"])
    df["abstract"] = df["abstract"].fillna("")
    df["text"] = df["title"] + " " + df["abstract"]
    return df.drop_duplicates(subset="news_id").reset_index(drop=True)

def load_behaviors(path):
    df = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS, usecols=["user_id","time","history"])
    return df.dropna(subset=["history"]).reset_index(drop=True)

def behaviors_to_interactions(df):
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Expanding histories"):
        for article_id in str(row["history"]).strip().split():
            rows.append({VIEW_TIME: row["time"], VIEW_USER: row["user_id"], VIEW_ARTICLE: article_id})
    return pd.DataFrame(rows)

def generate_embeddings(news_df, model_name="all-MiniLM-L6-v2", batch_size=256):
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    texts  = news_df["text"].tolist()
    ids    = news_df["news_id"].tolist()
    print(f"Embedding {len(texts)} articles...")
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return pd.DataFrame({EMB_ID_COL: ids, EMB_VEC_COL: [str(v.tolist()) for v in vectors]})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--dev_dir",   required=True)
    parser.add_argument("--out_dir",   default="data/raw")
    parser.add_argument("--model",     default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading news.tsv files...")
    all_news = pd.concat([load_news(os.path.join(args.train_dir,"news.tsv")),
                          load_news(os.path.join(args.dev_dir,"news.tsv"))], ignore_index=True)
    all_news = all_news.drop_duplicates(subset="news_id").reset_index(drop=True)
    print(f"  Total unique articles: {len(all_news):,}")

    emb_df  = generate_embeddings(all_news, args.model, args.batch_size)
    emb_out = os.path.join(args.out_dir, "mind_embeddings.csv")
    emb_df.to_csv(emb_out, index=False)
    print(f"\n✅ Saved embeddings → {emb_out}  ({len(emb_df):,} rows)")

    print("\nLoading behaviors.tsv files...")
    all_beh = pd.concat([load_behaviors(os.path.join(args.train_dir,"behaviors.tsv")),
                         load_behaviors(os.path.join(args.dev_dir,"behaviors.tsv"))], ignore_index=True)
    print(f"  Total behavior rows: {len(all_beh):,}")

    views_df  = behaviors_to_interactions(all_beh)
    valid_ids = set(emb_df[EMB_ID_COL].tolist())
    views_df  = views_df[views_df[VIEW_ARTICLE].isin(valid_ids)]
    views_out = os.path.join(args.out_dir, "mind_views.csv")
    views_df.to_csv(views_out, index=False)
    print(f"✅ Saved interactions → {views_out}  ({len(views_df):,} rows)")

    print(f"\n── Summary ──────────────────────────────")
    print(f"  Articles : {len(emb_df):,}")
    print(f"  Users    : {views_df[VIEW_USER].nunique():,}")
    print(f"  Interactions: {len(views_df):,}")
    print(f"\nNext step:")
    print(f"  python -m scripts.prep_processed --dataset NU --emb_csv {emb_out} --views_csv {views_out} --topk 25")

if __name__ == "__main__":
    main()
