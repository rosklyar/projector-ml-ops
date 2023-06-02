import logging
import os
import io
from PIL import Image
from typing import List

from serving.predictor import Predictor

logger = logging.getLogger()


class SeldonAPI:
    
    def __init__(self):
        self.predictor = Predictor.default_from_model_registry(os.getenv("MODEL_ID"), os.getenv("MODEL_PATH"))

    def predict(self, X, features_names: List[str]):
        logger.info(f"Predicting on {X}")
        result = self.predictor.predict(X)
        logger.info(f"Predicted result: {result}")
        return result