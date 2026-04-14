````markdown
# Fidelity Metrics for Explainable Recommenders – NU (COSINE Recommender)

This repository contains the main codebase for the empirical work of the thesis **"Refining Fidelity Metrics for Explainable Recommendations"** (VU) in collaboration with Mediahuis.

It builds on and extends an existing framework for explainable recommenders, and adapts it to

- Use a **news-style (NU)** dataset
- Use a **content-based COSINE similarity recommender** as the primary model
- Support multiple **explainers** and **fidelity metrics**, plus confidence intervals and diagnostics

It allows you to

- Run a **content-based COSINE recommender** on your own dataset
- Construct or train multiple explainers
  - **Jaccard-based**
  - **Cosine-based**
  - **LIME**
  - **ACCENT**
  - **LXR**
  - **SHAP** (cluster-based surrogate)
- Evaluate these explainers using **fidelity-style metrics** (DEL/INS, NDCG, POS@K),
  optionally with **bootstrap confidence intervals**, and run **recommender diagnostics**

This guide explains, step by step

1. How to set up the Python environment
2. How to prepare and store your dataset
3. How to configure the project
4. How to run the COSINE recommender with all explainers
5. How to evaluate explainers and diagnostics
6. How to reuse existing models and skip training steps

> **Scope note**  
> The thesis results are based on the **COSINE recommender only**.  
> Other recommenders (e.g. MLP) are optional and not part of the reported experiments.

---

## Two components in this repository

- Offline explanation pipeline  
  This is the main codebase and reproduces the thesis experiments

- Streamlit user study prototype  
  This is a separate interactive tool built after the pipeline work  
  It does not run the offline evaluation pipeline

---

## 1. Repository Structure

After cleanup, the relevant parts of the repository look like this

```text
.
├── src/
│   ├── config.py                 # Project-wide configuration (paths, checkpoint dictionaries)
│   ├── data_processing.py        # Generic data processing utilities (legacy datasets)
│   ├── explainers.py             # Implementation of explainer models (LXR, ACCENT, etc.)
│   ├── lime.py                   # LIME implementation
│   ├── metrics.py                # Implementation of explainer evaluation metrics
│   ├── models.py                 # Architectures of recommender models (MLP, VAE, NCF, etc.)
│   ├── nu_data.py                # NU-specific data loading and splitting
│   ├── utils.py                  # Utility functions (sampling, ranking, logging, etc.)
│   └── visualization.py          # Code for generating plots for the results
├── scripts/
│   ├── prep_processed.py         # Prepare NU processed data and similarity dictionaries
│   ├── train_lxr.py              # Train the LXR explainer for a given recommender
│   ├── make_item_clusters.py     # Cluster items based on embeddings
│   ├── make_shap_values.py       # Build SHAP-like surrogate values per user and cluster
│   ├── build_explanation_dicts.py# Precompute explanation dictionaries for all methods
│   ├── evaluate.py               # Evaluate explainer fidelity metrics (DEL/INS, NDCG, POS@K)
│   ├── eval_confint.py           # Compute confidence intervals over explainer metrics
│   └── eval_recsys_diagnostics.py# Evaluate recommender quality (HR, NDCG, MRR, etc.)
├── notebooks/
│   ├── Data_preparation_Explanation_Pipeline.ipynb  # Legacy notebook version of Step 0 style preprocessing
│   ├── Data Analysis & preparation.ipynb            # Creates Streamlit input files (separate from pipeline)
│   ├── Streamlit_UserStudy.ipynb                    # Streamlit prototype notebook
│   ├── LXR_training.ipynb                           # Interactive LXR training experiments (optional)
│   ├── SHAP_MLP_clusters.ipynb                      # Legacy SHAP analysis notebooks (not needed for COSINE)
│   ├── SHAP_NCF_clusters.ipynb
│   ├── SHAP_VAE_clusters.ipynb
│   ├── lime.ipynb                                   # LIME explanation examples (optional)
│   ├── metrics_discrete.ipynb                       # Discrete metric examples (optional)
│   ├── metrics_continous.ipynb                      # Continuous metric examples (optional)
│   └── visualization.ipynb                          # Plotting and result inspection (optional)
├── data/
│   ├── raw/                      # Place your own embeddings and interactions CSVs here
│   └── processed/
│       └── NU/                   # Automatically generated NU processed data and artifacts
├── checkpoints/                  # Explainer checkpoints (.pt), e.g. LXR
├── results/                      # Experiment results (tables, plots, confidence intervals)
└── logs/                         # Training logs (optional)
```

---

## 2. Environment Setup

This section explains which Python version and packages to use, and how to create an isolated environment.

Experiments for this thesis were run with

- **Python 3.11.3**

The code should work with **Python ≥ 3.10**, but for best reproducibility, use **Python 3.11.x**.

Check your Python version

```bash
python --version
```

If you get a different version (e.g. `3.12.x`), the code may still work, but minor differences in behaviour are possible.

To avoid conflicts with other projects, create a virtual environment in the project root

```bash
python -m venv .venv

# Activate it

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux or macOS
source .venv/bin/activate

pip install --upgrade pip
```

---

## 3. Installing Dependencies

All required package versions are stored in

- `requirements_project.txt`

Install them with

```bash
pip install -r requirements_project.txt
```

---

## 4. Data Preparation

This section explains what your dataset should look like and where to store it.

> **Important**
> The pipeline does not generate embeddings for you.
> You must embed your items yourself first, then store them in the exact column names described below.
> If you change those names, the code will break unless you also modify `nu_data.py`.

You must provide two CSV files

1. An embeddings CSV with one row per item or article
2. An interactions CSV with one row per user item interaction

Both go in `data/raw/`.

---

### 4.1. Embeddings CSV (`data/raw/my_embeddings.csv`)

Required columns

| Column                    | Type          | Description                          |
| ------------------------- | ------------- | ------------------------------------ |
| `SOURCE_SHORT_ARTICLE_ID` | string or int | Unique ID for each item or article   |
| `embedding`               | string        | Embedding vector encoded as a string |

Accepted embedding formats

- `"[0.12, -0.03, 0.55, 0.11]"`
- `"[0.12 -0.03 0.55 0.11]"`
- `"0.12, -0.03, 0.55, 0.11"`
- `"0.12 -0.03 0.55 0.11"`

> **Strict requirement**
> If an `ARTICLE_ID` appears in your interactions CSV but does not appear as a `SOURCE_SHORT_ARTICLE_ID` with an embedding, that interaction is dropped.

---

### 4.2. Interactions CSV (`data/raw/my_views.csv`)

Required columns

| Column       | Type            | Description                                                       |
| ------------ | --------------- | ----------------------------------------------------------------- |
| `TIMESTAMP`  | sortable string | When the interaction occurred (parseable by `pandas.to_datetime`) |
| `USER_ID`    | string or int   | User identifier                                                   |
| `ARTICLE_ID` | string or int   | Item identifier                                                    |

Example

```csv
TIMESTAMP,USER_ID,ARTICLE_ID
2024-01-01 08:03:10,42,1234
2024-01-01 08:04:55,42,5678
2024-01-01 09:15:00,99,5678
```

---

### 4.3. Train, test, and static test creation

You do not manually create train test splits.

Internally (`nu_data.py`)

- Users with fewer than `min_profile` interactions are discarded
- For each remaining user
  - Interactions are sorted by time
  - Only the last `max_profile` interactions may be kept
  - The first part becomes train and the last part becomes test according to `split_ratio`

Use `scripts/prep_processed.py` to create processed files in `data/processed/NU/`

- `train_data_NU.csv`
- `test_data_NU.csv`
- `static_test_data_NU.csv`
- `pop_dict_NU.pkl`
- `jaccard_based_sim_NU.pkl` and `cosine_based_sim_NU.pkl`

> **Note on notebooks**
> `notebooks/Data_preparation_Explanation_Pipeline.ipynb` is a legacy notebook that contains helper code for preprocessing and splitting.
> For reproducible runs, prefer `scripts/prep_processed.py` because it also builds the similarity dictionaries required downstream.

---

## 5. Configuration (`src/config.py`)

Some scripts use `src/config.py` to know where checkpoints are and what model dimensions to expect.
For the COSINE recommender, the main thing you need is

- A correct project root and checkpoints directory
- A correct LXR checkpoint configuration for COSINE

---

## 6. From-Scratch Pipeline (COSINE Recommender)

All commands assume

- You run them from the project root
- Your virtual environment is active

Replace

- `data/raw/my_embeddings.csv` with your embeddings path
- `data/raw/my_views.csv` with your interactions path

You must use `--dataset NU` in all commands unless you extend the code.

---

### 6.1. STEP 0 – Preprocess NU and build similarity dictionaries

```bash
python -m scripts.prep_processed --dataset NU --emb_csv data/raw/my_embeddings.csv --views_csv data/raw/my_views.csv --topk 25
```

---

### 6.2. STEP 1 – Train LXR explainer for COSINE

```bash
python -m scripts.train_lxr --dataset NU --recommender COSINE --emb_csv data/raw/my_embeddings.csv --views_csv data/raw/my_views.csv --min_profile 5 --min_test 5 --split_ratio 0.8 --hidden_dim 128 --epochs 3 --lr 1e-3 --l1_lambda 1e-3 --bpr_weight 1.0 --out checkpoints/lxr_explainer_NU_COSINE.pt
```

---

### 6.3. STEP 2 – Build SHAP surrogate artifacts

#### 6.3.1. Cluster items

```bash
python -m scripts.make_item_clusters --emb_csv data/raw/my_embeddings.csv --views_csv data/raw/my_views.csv --clusters 300 --out_dir data/processed/NU
```

#### 6.3.2. Compute SHAP-like values per user

```bash
python -m scripts.make_shap_values --emb_csv data/raw/my_embeddings.csv --views_csv data/raw/my_views.csv --clusters 300 --out_dir data/processed/NU
```

---

### 6.4. STEP 3 – Build explanation dictionaries (COSINE)

```bash
python -m scripts.build_explanation_dicts --dataset NU --recommender COSINE --emb_csv data/raw/my_embeddings.csv --views_csv data/raw/my_views.csv --min_profile 5 --min_test 5 --split_ratio 0.8 --eval_users 0 --random_eval --seed 42 --methods jaccard,cosine,lime,accent,lxr,shap
```

---

### 6.5. STEP 4 – Evaluate explainer metrics (COSINE)

```bash
python -m scripts.evaluate --dataset NU --recommender COSINE --emb_csv data/raw/my_embeddings.csv --views_csv data/raw/my_views.csv --min_profile 5 --min_test 5 --split_ratio 0.8 --eval_users 100000 --random_eval --seed 42
```

---

### 6.6. STEP 5 – Confidence intervals (optional)

```bash
python -m scripts.eval_confint --dataset NU --recommender COSINE --emb_csv data/raw/my_embeddings.csv --views_csv data/raw/my_views.csv --eval_users 0 --methods jaccard,cosine,lime,accent,lxr,shap --steps 5 --B 1000
```

---

### 6.7. STEP 6 – Recommender diagnostics (optional)

```bash
python -m scripts.eval_recsys_diagnostics --dataset NU --recommender COSINE --emb_csv data/raw/my_embeddings.csv --views_csv data/raw/my_views.csv --eval_users 10000 --ks 5,10,20 --outdir results/diagnostics
```

---

## 7. Skip mode

If you already have

- Processed NU data in `data/processed/NU/`
- LXR checkpoint in `checkpoints/` if you want LXR
- SHAP artifacts if you want SHAP
- Explanation dicts in `data/processed/NU/COSINE_explanation_dicts.pkl`

Then you can skip Steps 0 to 3 and only run evaluation and optional extras.

---

## 8. Streamlit user study prototype

This prototype is separate from the offline pipeline.

### 8.1. Inputs

Put these files in the same folder as the Streamlit script

- `clean_meta_data_embeddings.csv`  
  Articles with embeddings used by the prototype recommender

- `example_users_with_ids.csv`  
  Optional predefined user profiles shown in the UI

### 8.2. Export the notebook to a Python file

From the project root

```bash
jupyter nbconvert --to script notebooks/Streamlit_UserStudy.ipynb
```

This creates `Streamlit_UserStudy.py`.

### 8.3. Run Streamlit

In a terminal, change directory to the folder where you saved the exported `.py` file and the CSV files

```bash
cd "PATH_TO_YOUR_FOLDER"
streamlit run Streamlit_UserStudy.py
```

The app writes interaction logs to `logs/events.csv` and creates the folder if needed.

````


## 9. Reproducing Thesis Experiments: Exact Parameters & Rationale

This section documents the **exact parameter values** used in the thesis experiments and explains **why** they were chosen.  
It ties directly to the scripts described earlier (similarity dictionaries, SHAP surrogate, LXR, explanation dicts, evaluation, CIs, diagnostics).

Full command list and run notes are also in:
[docs/experiments.md](docs/experiments.md).
---

### 9.1. Common settings (used across scripts)

These flags recur in several commands:

- `--dataset NU`  
  Use the NU news-style dataset (custom loader in `nu_data.py`).

- `--recommender COSINE`  
  Use the **content-based COSINE** recommender as the primary model (no neural training).

- `--emb_csv "…\\clean_meta_data_embeddings.csv"`  
  Path to the embeddings CSV (see Data Preparation). Contains `SOURCE_SHORT_ARTICLE_ID` and `embedding`.

- `--views_csv "…\\views_data.csv"`  
  Path to the interactions CSV (see Data Preparation). Contains `TIMESTAMP`, `USER_ID`, `ARTICLE_ID`.

- `--eval_users {100000, 10000, 1000}`  
  Three evaluation cohorts:
  - 100k users: main, most stable curves.
  - 10k users: reduced but still robust; used where compute limits are tighter.
  - 1k users: fast debugging / sanity checks.

- `--random_eval`  
  Always sample users **randomly**, not “first N”.  
  This avoids time-order or user-ID biases in the evaluation cohort.

- `--seed 42`  
  Fix random seed so results are reproducible and user subsets are **nested**:
  - The 1,000-user sample is a subset of the 10,000-user sample.
  - The 10,000-user sample is a subset of the 100,000-user sample.

This nesting property is very useful for comparing curves across scales.
```
---
```
### 9.2. Similarity dictionaries (Jaccard baseline)

**Script:** `scripts.build_sim_dicts`  
(Conceptually: see the section on similarity dictionaries / Jaccard explainer.)

**Command used in the thesis study:**

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.build_sim_dicts \
  --dataset NU \
  --emb_csv "…\clean_meta_data_embeddings.csv" \
  --views_csv "…\views_data.csv"
````

**What it produces:**

* `jaccard_based_sim_NU.pkl`
  Used by the **Jaccard-based explainer** and as a similarity baseline.

**Why this setup:**

* No extra tuning flags: Jaccard similarity is fully determined by the observed interaction matrix.
* You only need to rerun this if:

  * the interactions (`views_data.csv`) change, or
  * you change the filtering logic (e.g. `min_profile` in `nu_data.py`).

---
```
### 9.3. SHAP surrogate prerequisites

**Script:** `scripts.prepare_shap_surrogate`
(Conceptually: combines “cluster items” and “compute SHAP-like values”.)

**Command used in the thesis study:**

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.prepare_shap_surrogate --dataset NU --recommender COSINE --emb_csv "…\clean_meta_data_embeddings.csv" --views_csv "…\views_data.csv" --clusters 300
```

**Key parameter:**

* `--clusters 300`

  * You had about ~1.5k items; 300 clusters gives a **moderately fine granularity**.
  * Too few clusters → explanations become very coarse (many items in one cluster).
  * Too many clusters → noisy, sparse cluster signals and higher compute.

**What it produces:**

* `item_to_cluster.npy` – maps each item to one of 300 clusters.
* `shap_values.npy` – per-user per-cluster surrogate SHAP scores.

These are later consumed by `scripts.build_explanation_dicts --methods shap`.
```
---
```
### 9.4. LXR training (COSINE explainer)

**Script:** `scripts.train_lxr`
(See main README section on LXR training.)

**Command used in the thesis study:**

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.train_lxr --dataset NU --recommender COSINE --emb_csv "…\clean_meta_data_embeddings.csv" --views_csv "…\views_data.csv" --epochs 3 --hidden_dim 128 --lr 1e-3 --l1_lambda 1e-3 --bpr_weight 1.0 --out lxr_explainer_NU_COSINE.pt
```

**Key parameters and why:**

* `--epochs 3`

  * Short training run for a laptop-friendly runtime.
  * Enough to learn non-trivial masks without overfitting.

* `--hidden_dim 128`

  * Capacity of the LXR internal MLP.
  * 128 is a standard, lightweight size: expressive but not huge.

* `--lr 1e-3`

  * Typical learning rate for Adam-like optimisers; stable and does not require a schedule for these runs.

* `--l1_lambda 1e-3`

  * Encourages **sparse** explanations (fewer features with non-zero attribution).
  * Makes explanations easier to interpret and align with the idea of a “small set of important items”.

* `--bpr_weight 1.0`

  * Balances ranking-oriented loss (BPR) against reconstruction / regularisation.
  * A neutral starting point where BPR and other terms contribute comparably.

* `--out lxr_explainer_NU_COSINE.pt`

  * File name referenced in `LXR_checkpoint_dict` in `config.py` as:

    ```python
    LXR_checkpoint_dict = {
        ("NU", "COSINE"): ("lxr_explainer_NU_COSINE.pt", 128),
    }
    ```
  * Ensures later scripts can find the correct explainer + hidden_dim.

---

### 9.5. Build explanation dictionaries (per method)

**Script:** `scripts.build_explanation_dicts`
(Conceptual description: see the “Build Explanation Dictionaries” section.)

The thesis built the explanation dictionary file
`data/processed/NU/COSINE_explanation_dicts.pkl` **incrementally**, one method at a time:

* `lime`
* `accent`
* `lxr`
* `shap`

They all share:

* `--dataset NU`
* `--recommender COSINE`
* `--emb_csv` / `--views_csv` as above
* `--random_eval --seed 42`
* `--save_every 2000 --log_every 200`
  (So you get progress logs and partial checkpoints.)

Below are the exact commands and rationale for **each method**.

---

#### 9.5.1. LIME explanations

**Command used in the thesis study (large user superset):**

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.build_explanation_dicts --dataset NU --recommender COSINE --emb_csv "…\clean_meta_data_embeddings.csv" --views_csv "…\views_data.csv" --methods lime --eval_users 100000 --random_eval --seed 42 --save_every 2000 --log_every 200
```

**Why these values:**

* `--eval_users 100000`

  * LIME is expensive; you built explanations **once** on the largest cohort.
  * Because `--random_eval` + fixed `--seed` are used everywhere, that 100k cohort:

    * contains the 10k cohort as a subset,
    * contains the 1k cohort as a subset,
      enabling reuse.

* `--save_every 2000` / `--log_every 200`

  * Periodic saving avoids losing work if interrupted.
  * Logging every 200 users gives decent progress visibility without spamming.

**Internal LIME settings (in code, e.g. `evaluate.py`):**

You also set LIME’s perturbation hyperparameters in Python:

* `min_pert = 50`
* `max_pert = 100`
* `num_of_perturbations = 150`

**Why:**

* These numbers balance:

  * **Stability** of the linear surrogate (enough perturbations),
  * and **runtime** (still reasonable on a laptop).
* 150 perturbations total, with 50–100 features perturbed each time, gives a robust regression fit for tabular/masked-item LIME in this setting.

---

#### 9.5.2. ACCENT explanations

**Command used in the thesis study:**

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.build_explanation_dicts --dataset NU --recommender COSINE --emb_csv "…\clean_meta_data_embeddings.csv" --views_csv "…\views_data.csv" --methods accent --accent_topk 3 --eval_users 10000 --random_eval --seed 42 --save_every 2000 --log_every 200
```

**Key parameter:**

* `--accent_topk 3`

  * Controls how many items ACCENT highlights per explanation.
  * You deliberately chose a **small k = 3**:

    * Forces the method to concentrate its mass on very few items.
    * Makes different methods’ curves **more distinguishable** in DEL/INS.
    * Larger k (e.g., 5 or 10) tends to smooth curves and make them closer to popularity baselines.

**Eval users:**

* `--eval_users 10000`

  * 10k users is a good compromise:

    * Enough to stabilise metrics.
    * Less costly than 100k for a more involved explainer.

---

#### 9.5.3. LXR (dictionary construction, after training)

After training LXR in Section 9.4, you attached LXR explanations to users with:

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.build_explanation_dicts --dataset NU --recommender COSINE --emb_csv "…\clean_meta_data_embeddings.csv" --views_csv "…\views_data.csv" --methods lxr --eval_users 10000 --random_eval --seed 42 --save_every 2000 --log_every 200
```

**Why 10k users:**

* Similar reasoning as ACCENT:

  * LXR is fully learned but cheaper than LIME.
  * 10k users gives stable metrics while keeping runtimes manageable.

This step reads `lxr_explainer_NU_COSINE.pt` from `checkpoints/` (via `LXR_checkpoint_dict`) and stores per-user explanations in the shared `.pkl`.

---

#### 9.5.4. SHAP surrogate explanations

**Command used in the thesis study:**

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.build_explanation_dicts --dataset NU --recommender COSINE --emb_csv "…\clean_meta_data_embeddings.csv" --views_csv "…\views_data.csv" --methods shap --eval_users 10000 --random_eval --seed 42 --save_every 2000 --log_every 200
```

**Why:**

* Uses `item_to_cluster.npy` and `shap_values.npy` from `prepare_shap_surrogate`.
* 10k users again balances:

  * expressiveness / stability,
  * with runtime (especially since SHAP involves aggregations).

**Reuse note:**
Because all these explanation dicts are built with `--random_eval --seed 42`, you can consistently reuse them for evaluation at 1k/10k/100k scales, drawing nested subsets of users.

---

### 9.6. Evaluation: metrics and internal design choices

**Script:** `scripts.evaluate`
(Conceptual overview in the main README section on evaluation.)

**Command used for the main 100k-user evaluation:**

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.evaluate --dataset NU --recommender COSINE --emb_csv "…\clean_meta_data_embeddings.csv" --views_csv "…\views_data.csv" --eval_users 100000 --random_eval --seed 42
```

**Important internal design decisions (in code, not flags):**

* **Continuous/global DEL/INS curves**

  * The global masking variant goes from 0% to 100% removed/inserted.
  * The code is set up so **all methods share the same endpoint** at 100% mask:

    * This matches the original paper’s idea that, at full removal, all explanations converge.

* **Discrete DEL/INS curves (history-only)**

  * For discrete plots, the code uses **K = 1..5** historical items.
  * This matches the “Ke” setup in the original paper (small K histories).

* **Ranking behaviour:** `exclude_seen_eval = False`

  * When ranking, previously seen items are **not removed** from the candidate set.
  * This aligns with the baseline in the referenced framework where:

    * Endpoints of DEL/INS curves are comparable across methods,
    * And metrics like NDCG are computed over a consistent item universe.

* **Signed importance**

  * For methods that can produce negative contributions (LIME, LXR, SHAP), you **preserve sign** in the mask construction.
  * This allows those methods’ curves to diverge meaningfully (e.g. down-weighting harmful items), instead of everything collapsing to absolute scores.

Together, these settings ensure that:

* All methods start from the **same initial recommender performance**.
* All methods end at the **same 100% masked endpoint** in the continuous setting.
* Differences across curves are due to **ranking of importance**, not implementation artefacts.

---

### 9.7. Diagnostics: recommender quality

**Script:** `scripts.eval_recsys_diagnostics`
(Described conceptually in the “Recommender Diagnostics (Optional)” section.)

**Command used in the thesis study:**

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.eval_recsys_diagnostics --dataset NU --recommender COSINE --emb_csv "…\clean_meta_data_embeddings.csv" --views_csv "…\views_data.csv" --eval_users 100000 --ks 5,10,20
```

**Why these values:**

* `--eval_users 100000`

  * Use the largest cohort for the most stable estimates of HR@K, NDCG@K, MRR.

* `--ks 5,10,20`

  * Standard top-K cutoffs for recommender evaluation.
  * Give a good picture of:

    * Very top suggestions (K=5),
    * A more relaxed view (K=10),
    * A wider but still practical slate (K=20).

This command checks the **recommender itself**, independent of explainers.

---

### 9.8. Confidence intervals for explainer metrics

**Script:** `scripts.eval_confint`
(Conceptual explanation in the “Confidence Intervals” section.)

**Command used in the thesis study:**

```bash
$env:PYTHONUNBUFFERED="1"; python -u -m scripts.eval_confint --dataset NU --recommender COSINE --emb_csv "…\clean_meta_data_embeddings.csv" --views_csv "…\views_data.csv" --eval_users 0 --methods jaccard,cosine,lime,accent,lxr,shap --steps 5 --B 1000
```

**Key parameters and why:**

* `--eval_users 0`

  * `0` means: use **all users available** in the explanation dicts.
  * Maximises statistical power for bootstrap estimation.

* `--methods jaccard,cosine,lime,accent,lxr,shap`

  * Include all explainers used in the thesis.
  * If a method is missing (no dict), drop it from this list.

* `--steps 5`

  * Use 5 DEL/INS steps (e.g. 0%, 20%, 40%, 60%, 80%, 100%).
  * Enough to see the shape of the degradation curve without exploding compute.

* `--B 1000`

  * Number of bootstrap samples.
  * 1000 replicates yields **smooth, stable confidence intervals** while remaining tractable.

These bootstrapped confidence bands provide **uncertainty estimates** for each explainer’s DEL/INS / NDCG / POS@K curves, used in the thesis to support robustness claims.
```
---
```
## 10. Optional: Other Recommenders (Not Used in Thesis)

The underlying framework also supports **learned recommenders** (e.g. MLP) via:

```bash
python -m scripts.train --dataset NU --recommender MLP ...
```

> **Important**
> These models **were not part of the thesis experiments or reported results**.
> They are provided only as an optional extension for users who want to explore additional architectures.
> For thesis reproducibility, you only need the **COSINE** pipeline described above.

```


