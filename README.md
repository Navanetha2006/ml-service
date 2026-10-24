# Breast Cancer Prediction Service

A small ML project: train models, compare them, and serve the best one with FastAPI and a simple web page.

## Data
Dataset: scikit-learn `load_breast_cancer` (569 rows). I used 5 "mean" features to keep the UI small.
Target: 0 = malignant, 1 = benign. Checked missing values (none), duplicates and class balance (about 63% benign) in `train.py`.

## Avoiding data leakage
- Train/test split is done first (80/20, stratified, fixed seed).
- `StandardScaler` is inside a `Pipeline`, so it only learns from training data (also inside each CV fold).
- Models are compared using 5-fold CV on the training set only.
- The test set is used once, for the final model.
- Only features available before diagnosis are used. The target column is never a feature.

## Model comparison
Logistic Regression, KNN, Random Forest. Chosen by mean CV F1. Results are saved to `artifacts/comparison.csv`
and final test metrics to `artifacts/metadata.json`. I used F1 (not just accuracy) because both
false positives and false negatives matter here.

## Run locally
```
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```
Open http://localhost:8000 for the UI, http://localhost:8000/docs for the API docs.

## API
| Endpoint | Description |
|---|---|
| GET /health | service is up |
| GET /model-info | model name, features, metrics |
| POST /predict | returns label and probability |

Error codes: 422 for bad input, 503 if artifacts are missing or the model is not compatible
(different scikit-learn version or different features).

## Tests
`pytest -v` covers valid requests, invalid inputs, missing artifacts and model compatibility.

## Docker
```
docker build -t ml-service .
docker run -p 8000:8000 ml-service
```
The model is trained during the build, so the model and scikit-learn versions always match.

## CI/CD
GitHub Actions (`.github/workflows/ci.yml`): train, run tests, build the Docker image, start the container
and check `/health`, `/predict` and the UI with curl. On a tag like `v1.0` the image is pushed to GitHub Container Registry.

## Limitations
This is a student project on a toy dataset. It is not a medical tool.
