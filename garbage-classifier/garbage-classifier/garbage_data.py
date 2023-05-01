from torchvision import torch, transforms
from torchvision.datasets import ImageFolder
from transformers import BeitFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class FeatureExtractor(torch.nn.Module):
    
    def __init__(self, model_name):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(model_name)
        
    def forward(self, image):
        return self.feature_extractor(images=image, return_tensors="pt")

    def __repr__(self):
        return self.__class__.__name__ + '()'

class LazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)