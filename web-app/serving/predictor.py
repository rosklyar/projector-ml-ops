import logging
from json import load
from pathlib import Path

import torch
import wandb
from filelock import FileLock

from transformers import BeitFeatureExtractor

logger = logging.getLogger()

MODEL_LOCK = ".lock-file"


def load_from_registry(model_name: str, model_path: Path):
    with wandb.init(project="garbage-classifier") as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")


class Predictor:

    def __init__(self, model_load_path: str):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._encoder = FeatureExtractor(
            "microsoft/beit-base-patch16-224-pt22k-ft22k")
        
        self._model = torch.load(
            Path(model_load_path) / "model.pth", map_location=self._device)
        self._model.eval()
        
        config_path = Path(model_load_path) / "config.json"
        json_config = load(config_path.open(encoding="utf-8"))
        self._classes = json_config["labels"]

    @torch.no_grad()
    def predict(self, image):
        image_encoded = self._encoder(image)['pixel_values'].to(self._device)
        output = self._model(image_encoded).logits
        predicted = output.argmax(dim=-1)
        predicted_class = str(predicted.item() + 1)
    
        if predicted_class not in self._classes:
            raise KeyError(f"Model and its classes config mismatched. Predicted class={predicted_class}. Classes={str(self._classes)}")
    
        return self._classes[predicted_class]

    @classmethod
    def default_from_model_registry(cls, model_id, model_path) -> "Predictor":
        with FileLock(MODEL_LOCK):
            if not (Path(model_path) / "model.pth").exists():
                load_from_registry(model_name=model_id, model_path=model_path)

        return cls(model_load_path=model_path)
    
    def get_classes_config(self):
        return self._classes
    
class FeatureExtractor(torch.nn.Module):
    
    def __init__(self, model_name):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(model_name)
        
    def forward(self, image):
        return self.feature_extractor(images=image, return_tensors="pt")

    def __repr__(self):
        return self.__class__.__name__ + '()'
