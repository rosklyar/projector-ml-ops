import pytest
import torch
from garbage_classifier import trainer as t, model as m
from garbage_classifier import garbage_data as gd


@pytest.fixture
def data():
    return gd.GarbageData(
        "garbage_classifier/tests/data", 2, 0.2)

def test_get_optimizer():
    model = m.get_model(0.1, 512, 6)
    optimizer = t.get_optimizer(model, 'adam', 0.001)
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Adam)
    optimizer = t.get_optimizer(model, 'sgd', 0.001)
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.SGD)

def test_train_epoch(data: gd.GarbageData):
    model = m.get_model(0.1, 512, 6)
    optimizer = t.get_optimizer(model, 'adam', 0.001)
    train_dataloader = data.get_train_loader()
    avg_loss = t.train_epoch(model, train_dataloader, optimizer)
    assert avg_loss >= 0
    assert isinstance(avg_loss, float)