# Two-Stage Recommendation System (LightGCN Retrieval + DeepFM Ranking)

## Problem Statement
The platform has millions of historical ratings and tags. A popularity-only recommender under-personalizes the feed. Build a two-stage recommendation system (retrieval + ranking) with production-style MLOps and reproducible evaluation.

## Dataset
MovieLens 32M ratings + tags (public). The pipeline supports time-based splits and sampling for local runs.

## Solution Overview
1. Retrieval (high-recall): LightGCN generates top-N candidates per user.
2. Ranking (high-precision): DeepFM re-scores candidates with engineered features.
3. MLOps: MLflow for experiment tracking and artifact management.

## Tech Stack
Python, PyTorch, Pandas, NumPy, SciPy, MLflow, LightGCN, DeepFM, Jupyter, PyArrow

## Workflow (Before / During / After Training)
1. Preprocessing
   - De-duplicate interactions, filter invalid ratings, enforce min user/item interactions.
   - Time-based train/val/test split to prevent leakage.
   - Tag features are time-filtered to the training window.
2. Training
   - LightGCN for candidate generation.
   - DeepFM ranker with feature scaling, class-imbalance weighting, and early stopping.
3. Evaluation & Serving
   - Offline ranking metrics (Precision@K, Recall@K, NDCG@K, MAP@K, Coverage@K).
   - Batch generation of top-K recommendations.
   - Metrics saved to `data/processed/evaluation_metrics.json`.

## Results (Sample Run, K=10)
- Retrieval: Precision 0.0548, Recall 0.0569, NDCG 0.1763, MAP 0.0938
- Ranker: Precision 0.0597, Recall 0.0605, NDCG 0.2007, MAP 0.0828

## Learnings & Impact
- Built a production-style two-stage recommender with measurable ranking lift.
- Implemented leakage-aware preprocessing and evaluation to improve metric reliability.
- Applied feature engineering, imbalance handling, and MLOps tracking with MLflow.

## Project Structure
- `movielens_data/` raw data (already provided)
- `data/processed/` train/val/test splits and candidates
- `models/` trained artifacts
- `pipelines/` pipeline scripts
- `src/recsys/` core modules
- `tests/` minimal tests

## Quickstart
```bash
pip install -r requirements.txt
make prepare
make retrieval
make candidates
make ranker
make evaluate
make batch
```

## MLflow
```bash
mlflow ui --backend-store-uri mlruns
```

## EDA
- Notebook: `notebooks/01_eda.ipynb`

## Notes
- Default sampling limits to 5,000 users for manageable local runs.
- Change sampling and model config in `src/recsys/config.py`.
- Use `data/processed/summary.json` to verify split sizes.
