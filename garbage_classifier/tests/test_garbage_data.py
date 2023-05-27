import torch
import pytest

from garbage_classifier import garbage_data as gd


@pytest.mark.parametrize("input_size", [(3, 256, 256), (3, 512, 512), (3, 1024, 1024), (3, 640, 480)])
def test_feature_extractor_output_shape(input_size):
    feature_extractor = gd.FeatureExtractor(
        'microsoft/beit-base-patch16-224-pt22k-ft22k')
    assert feature_extractor is not None

    features = feature_extractor.forward(torch.rand(*input_size))
    assert features['pixel_values'].shape == torch.Size([1, 3, 224, 224])


def test_dataset_split_routine():
    batch_size = 2
    garbage_data = gd.GarbageData(
        "garbage_classifier/tests/data", "garbage_classifier/tests/data", batch_size)
    assert garbage_data is not None
    train_dataloader = garbage_data.get_train_loader()
    assert train_dataloader is not None
    assert len(train_dataloader) == 10 / batch_size
    val_dataloader = garbage_data.get_test_loader()
    assert val_dataloader is not None
    assert len(val_dataloader) == 10 / batch_size
