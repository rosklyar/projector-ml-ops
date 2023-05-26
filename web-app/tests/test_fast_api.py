import os
import pytest
from pathlib import Path 
from json import load
import shutil

from fastapi.testclient import TestClient
from serving.fast_api import app

client = TestClient(app)

def remove_model_path():
    model_path = os.getenv("MODEL_PATH")
    shutil.rmtree(model_path, ignore_errors=True)

@pytest.fixture(autouse=True)
def teardown(request):
    request.addfinalizer(remove_model_path)

def test_get_config():
    response = client.get("/config")
    assert response.status_code == 200
    assert response.json() == load((Path(os.getenv("MODEL_PATH")) / "config.json").open(encoding="utf-8"))["labels"]

def test_predict():
    with open("web-app/tests/assets/image.jpg", "rb") as image_file:
        response = client.post("/predict", files={"image_file": image_file})
        assert response.status_code == 200
        assert response.json() == {"result": "[PET 1] Пляшка прозора з-під напоїв з блакитним відтінком "}