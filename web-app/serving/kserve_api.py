import os
from typing import Dict
from serving.predictor import Predictor
from kserve import Model, ModelServer

class UwgCLassifierModel(Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.load()

    def load(self):
        self.predictor = Predictor.default_from_model_registry(os.getenv("MODEL_ID"), os.getenv("MODEL_PATH"))

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> str:
        image_data = payload["data"]
        print(f"Predicting on {image_data}")
        return self.predictor.predict(image_data)

if __name__ == "__main__":
    model = UwgCLassifierModel("uwg-classifier-model")
    ModelServer().start([model])