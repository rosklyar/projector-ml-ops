import os
import torch

class Config:

    def __init__(self):
        self.dataset_dir = 'data/kaggle-ds'
        self.checkpoint_dir = 'model/checkpoints'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # number of subfolders in dataset_dir
        self.classes = len(os.listdir(self.dataset_dir))

config = Config()
os.makedirs(os.path.join(config.checkpoint_dir, 'weights/latest'), exist_ok=True)