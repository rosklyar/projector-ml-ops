import pytest
import torch

from garbage_classifier import model_utils as m
from garbage_classifier import garbage_data as gd


@pytest.fixture
def garbage_data():
    return gd.GarbageData(
        "garbage_classifier/tests/data", 2, 0.2)

def test_model_creation():
    model = m.get_model('microsoft/beit-base-patch16-224-pt22k-ft22k', 0.1, 512, 6)
    prediction = model(torch.rand(1, 3, 224, 224))
    assert prediction.logits.shape == torch.Size([1, 6])

def test_model_scoring(garbage_data: gd.GarbageData):
    model = m.get_model('microsoft/beit-base-patch16-224-pt22k-ft22k', 0.1, 512, 6)
    score = m.score_model(model, garbage_data.get_val_loader())
    assert score >= 0   
