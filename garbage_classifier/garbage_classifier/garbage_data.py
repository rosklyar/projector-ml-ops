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
    
class GarbageData():

    def __init__(self, dataset_path, batch_size, val_split=0.2):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.feature_extractor = FeatureExtractor('microsoft/beit-base-patch16-224-pt22k-ft22k')
        data_transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
            self.feature_extractor])
        self.data = ImageFolder(self.dataset_path)
        train_data, validation_data = random_split(self.data, [int(len(self.data) * (1 - val_split)), int(len(self.data) * val_split)])
        self._train_loader = DataLoader(LazyDataset(train_data, data_transform_train), batch_size=self.batch_size, shuffle=True, num_workers=4)
        self._val_loader = DataLoader(LazyDataset(validation_data, transforms.Compose([self.feature_extractor])), batch_size=self.batch_size, shuffle=False, num_workers=4)

    def get_train_loader(self):
        return self._train_loader

    def get_val_loader(self):
        return self._val_loader