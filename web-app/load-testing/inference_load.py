import numpy as np
import os
import random
from pathlib import Path
from locust import HttpUser, between, task

images_pool = [
    "garbage_classifier/tests/data/paper/paper1.jpg",
    "garbage_classifier/tests/data/paper/paper2.jpg",
    "garbage_classifier/tests/data/paper/paper3.jpg",
    "garbage_classifier/tests/data/paper/paper4.jpg",
    "garbage_classifier/tests/data/paper/paper5.jpg",
    "garbage_classifier/tests/data/paper/paper6.jpg",
    "garbage_classifier/tests/data/plastic/plastic1.jpg",
    "garbage_classifier/tests/data/plastic/plastic2.jpg",
    "garbage_classifier/tests/data/plastic/plastic3.jpg",
    "garbage_classifier/tests/data/plastic/plastic4.jpg"
]


class PredictUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        image_path = Path(os.getcwd()) / random.choice(images_pool)
        with open(image_path, 'rb') as image_file:
            response = self.client.post(
                "/predict",
                files={"image_file": ("filename", image_file, "image/jpeg")},
                headers={"accept": "application/json"}
            )