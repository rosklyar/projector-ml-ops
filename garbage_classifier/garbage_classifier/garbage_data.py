import os
import tarfile
import boto3
import numpy as np
import shutil
from torchvision import torch, transforms
from torchvision.datasets import ImageFolder
from transformers import BeitFeatureExtractor
from torch.utils.data import Dataset, DataLoader

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

    def __init__(self, train_path, test_path, batch_size):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.feature_extractor = FeatureExtractor('microsoft/beit-base-patch16-224-pt22k-ft22k')
        data_transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
            self.feature_extractor])
        self.train_data = ImageFolder(self.train_path)
        self.test_data = ImageFolder(self.test_path)
        self._train_loader = DataLoader(LazyDataset(self.train_data, data_transform_train), batch_size=self.batch_size, shuffle=True, num_workers=4)
        self._test_loader = DataLoader(LazyDataset(self.test_data, transforms.Compose([self.feature_extractor])), batch_size=self.batch_size, shuffle=False, num_workers=4)

    def get_train_loader(self):
        return self._train_loader

    def get_test_loader(self):
        return self._test_loader
    
def load_train_data(s3_access_key, s3_secret_key, s3_bucket, s3_prefix):
    s3_client = boto3.client('s3', aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)
    folder = 'data/real-ds'
    _download_files_from_s3_bucket_folder(s3_client, s3_bucket, s3_prefix, f'{folder}/input')
    _create_train_test_split(f'{folder}/input', f'{folder}/train', f'{folder}/test')
    _create_tar_gz_folder(f'{folder}/train', 'data/real-ds/train.tar.gz')
    _create_tar_gz_folder(f'{folder}/test', 'data/real-ds/test.tar.gz')

def load_data(s3_access_key, s3_secret_key, s3_bucket, s3_prefix):
    s3_client = boto3.client('s3', aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key)
    folder = 'data/tmp'
    _download_files_from_s3_bucket_folder(s3_client, s3_bucket, s3_prefix, f'{folder}')
    _create_tar_gz_folder(f'{folder}', 'data/data.tar.gz')
    
def _create_train_test_split(src_folder, train_folder, test_folder):
    class_folders = [f for f in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, f))]

    for class_folder in class_folders:
        src_class_folder = os.path.join(src_folder, class_folder)
        train_class_folder = os.path.join(train_folder, class_folder)
        test_class_folder = os.path.join(test_folder, class_folder)

        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(test_class_folder, exist_ok=True)

        images = [f for f in os.listdir(src_class_folder) if os.path.isfile(os.path.join(src_class_folder, f))]
        np.random.shuffle(images)
        split_index = int(len(images) * 0.8)

        train_images = images[:split_index]
        test_images = images[split_index:]

        for img in train_images:
            shutil.copy(os.path.join(src_class_folder, img), os.path.join(train_class_folder, img))

        for img in test_images:
            shutil.copy(os.path.join(src_class_folder, img), os.path.join(test_class_folder, img))

def _create_tar_gz_folder(src_folder, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        for entry in os.scandir(src_folder):
            if entry.is_dir():
                arcname = os.path.basename(entry.path)
                tar.add(entry.path, arcname=arcname)


def _download_files_from_s3_bucket_folder(s3, bucket_name, folder_path, local_destination):
    paginator = s3.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name, 'Prefix': folder_path}

    for page in paginator.paginate(**operation_parameters):
        for item in page.get('Contents', []):
            if not item['Key'].endswith('/'):
                file_key = item['Key']
                local_file_path = os.path.join(local_destination, os.path.relpath(file_key, folder_path))

                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                print(f'Downloading {file_key} to {local_file_path}')
                s3.download_file(bucket_name, file_key, local_file_path)
                print(f'Downloaded {file_key} to {local_file_path}')

def extract_tar_gz(archive_path):
    if not os.path.exists(archive_path):
        print(f"The file '{archive_path}' does not exist.")
        return None

    if not tarfile.is_tarfile(archive_path):
        print(f"The file '{archive_path}' is not a valid tar archive.")
        return None

    # Create a unique directory for the extracted files
    extracted_folder = os.path.splitext(
        os.path.splitext(os.path.basename(archive_path))[0])[0]
    extracted_path = os.path.join(
        os.path.dirname(archive_path), extracted_folder)

    if not os.path.exists(extracted_path):
        os.makedirs(extracted_path)

    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extracted_path)
            print(
                f"Successfully extracted '{archive_path}' to '{extracted_path}'.")
            return extracted_path
    except Exception as e:
        print(f"Error while extracting '{archive_path}': {e}")
        return None
    