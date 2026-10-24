import json

import pytest
from fastapi.testclient import TestClient

import train
from app.main import app

client = TestClient(app)

VALID = {
    "mean_radius": 14.1,
    "mean_texture": 19.3,
    "mean_perimeter": 92.0,
    "mean_area": 654.0,
    "mean_smoothness": 0.096,
}


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    train.run(d)
    return d


@pytest.fixture
def with_model(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))


# ---------- valid requests ----------
def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ui_page_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "Predictor" in r.text


def test_predict_valid(with_model):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("malignant", "benign")
    assert 0 <= body["probability"] <= 1


def test_model_info(with_model):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["features"] == list(VALID.keys())


# ---------- invalid inputs ----------
def test_missing_field(with_model):
    bad = dict(VALID)
    del bad["mean_area"]
    assert client.post("/predict", json=bad).status_code == 422


def test_wrong_type(with_model):
    bad = dict(VALID, mean_radius="big")
    assert client.post("/predict", json=bad).status_code == 422


def test_negative_value(with_model):
    bad = dict(VALID, mean_radius=-5)
    assert client.post("/predict", json=bad).status_code == 422


def test_out_of_range(with_model):
    bad = dict(VALID, mean_smoothness=7)
    assert client.post("/predict", json=bad).status_code == 422


def test_empty_body(with_model):
    assert client.post("/predict", json={}).status_code == 422


# ---------- missing artifacts ----------
def test_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))  # empty folder
    r = client.post("/predict", json=VALID)
    assert r.status_code == 503
    assert "not found" in r.json()["detail"]


def test_missing_model_file_only(model_dir, tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text((model_dir / "metadata.json").read_text())
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert client.post("/predict", json=VALID).status_code == 503


# ---------- model compatibility ----------
def _copy_model(src, dst):
    (dst / "model.joblib").write_bytes((src / "model.joblib").read_bytes())
    return json.loads((src / "metadata.json").read_text())


def test_wrong_sklearn_version(model_dir, tmp_path, monkeypatch):
    meta = _copy_model(model_dir, tmp_path)
    meta["sklearn_version"] = "0.0.1"
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    r = client.post("/predict", json=VALID)
    assert r.status_code == 503
    assert "scikit-learn" in r.json()["detail"]


def test_wrong_features(model_dir, tmp_path, monkeypatch):
    meta = _copy_model(model_dir, tmp_path)
    meta["features"] = ["a", "b"]
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert client.post("/predict", json=VALID).status_code == 503


def test_corrupted_model_file(model_dir, tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text((model_dir / "metadata.json").read_text())
    (tmp_path / "model.joblib").write_bytes(b"not a model")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert client.post("/predict", json=VALID).status_code == 503
