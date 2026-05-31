# Churn MLOps Pipeline

A production-style batch churn scoring pipeline for a subscription business.
Validates data, trains a model, tracks experiments with MLflow, and writes
timestamped predictions — all in a single command.

---

## Architecture

```
Daily CSV
    │
    ▼
Data Validation ──── fail early with clear error
    │
    ▼
Model exists? ──No──► Train + MLflow Tracking + Save versioned .pkl
    │ Yes
    ▼
Batch Scoring
    │
    ▼
predictions/predictions_YYYYMMDD_HHMMSS.csv
```

---

## Project Structure

```
churn-mlops/
├── configs/
│   └── config.yaml          # All tuneable settings
├── data/
│   ├── raw/
│   │   ├── customers.csv        # Historical training data
│   │   └── customers_daily.csv  # Daily scoring input
│   ├── processed/
│   └── predictions/             # Output written here
├── models/                      # Versioned .pkl artifacts
├── mlruns/                      # MLflow tracking store
├── logs/                        # pipeline.log written here
├── src/
│   ├── utils.py       # Config, logging, path helpers
│   ├── validate.py    # Data validation layer
│   ├── train.py       # Model training + MLflow logging
│   ├── score.py       # Batch scoring
│   └── pipeline.py    # Orchestrator (runs all steps)
├── tests/
│   ├── test_validation.py
│   └── test_training.py
├── Dockerfile
└── requirements.txt
```

---

## Phase 1 — Run Locally

### 1. Set up environment

```bash
cd churn-mlops
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python src/pipeline.py
```

What happens:
1. Validates `data/raw/customers.csv`
2. Trains a RandomForestClassifier if no model exists
3. Validates `data/raw/customers_daily.csv`
4. Writes predictions to `data/predictions/predictions_<timestamp>.csv`
5. Logs everything to `logs/pipeline.log`

### 3. View MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

### 4. Run individual steps

```bash
python src/validate.py   # validate training data only
python src/train.py      # train and save model
python src/score.py      # score daily file
```

### 5. Run tests

```bash
pytest tests/ -v
```

### 6. Test early failure (break a column type)

Edit `data/raw/customers.csv` and change a `login_count` value to `"abc"`.
Re-run the pipeline — it will exit immediately with:

```
ValidationError: Type mismatch(es):
  Column 'login_count': expected int, got object
```

---

## Phase 2 — Google Cloud Deployment

### Prerequisites

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2a. Create a GCS bucket

```bash
gcloud storage buckets create gs://YOUR_BUCKET_NAME \
    --location=us-central1

# Upload training data
gcloud storage cp data/raw/customers.csv gs://YOUR_BUCKET_NAME/raw/
```

### 2b. Build and push Docker image

```bash
# Enable Artifact Registry
gcloud services enable artifactregistry.googleapis.com

# Create repository
gcloud artifacts repositories create churn \
    --repository-format=docker \
    --location=us-central1

# Build and push
clear
```

### 2c. Create a Cloud Run Job

```bash
gcloud run jobs create churn-job \
    --image us-central1-docker.pkg.dev/project-4b1192a0-e65c-4cde-a6a/churn/churn-pipeline:latest \
    --region us-central1 \
    --memory 512MiB

# Test run
gcloud run jobs execute churn-job --region us-central1
```

### 2d. Schedule nightly runs (1 AM UTC)

```bash
gcloud scheduler jobs create http nightly-churn \
    --schedule="0 1 * * *" \
    --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/YOUR_PROJECT_ID/jobs/churn-job:run" \
    --message-body="{}" \
    --oauth-service-account-email=YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --location=us-central1
```

---

## Configuration

All settings live in `configs/config.yaml`:

| Key | Description |
|-----|-------------|
| `model.n_estimators` | Number of trees in the forest |
| `model.test_size` | Fraction of data held out for evaluation |
| `model.random_state` | Seed for reproducibility |
| `features` | List of feature columns used for training and scoring |
| `paths.raw_data` | Path to historical training CSV |
| `paths.daily_data` | Path to daily scoring CSV |
| `paths.model_dir` | Directory where versioned models are saved |
| `paths.prediction_dir` | Directory where prediction CSVs are written |
| `mlflow.experiment_name` | MLflow experiment name |
| `gcp.*` | GCP project, bucket, and region (Phase 2) |

---

## Reproducibility

- `model.random_state: 42` in config ensures identical splits and model weights.
- Every training run saves a timestamped artifact (`model_YYYYMMDD_HHMMSS.pkl`).
- MLflow logs params, metrics, and the model artifact for every run.
- Re-running with the same config and data produces the same metrics.

---

## Done Criteria

| Check | How to verify |
|-------|--------------|
| Pipeline fails early on bad data | Break a column type → see `ValidationError` |
| Reproducible training | Run `train.py` twice with same config → same metrics in MLflow |
| Versioned artifacts | Check `models/` — each run adds a new timestamped `.pkl` |
| Predictions written | Check `data/predictions/` after scoring |
| Logs available | Check `logs/pipeline.log` |
