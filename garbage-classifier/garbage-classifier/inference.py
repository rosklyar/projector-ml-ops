from concurrent.futures import ProcessPoolExecutor
from torchvision import torch, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from garbage_data import LazyDataset, FeatureExtractor
from tqdm import tqdm
import time

BATCH_SIZE = 16
PROCESS_COUNT = 8

def load_model():
    model = torch.load("model/model.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

def predict(model, batch):
    frames = torch.squeeze(batch[0]['pixel_values'])
    predicted = model(frames)
    return predicted

def single_process_prediction(model, feature_extractor, dataset: ImageFolder):
    dataloader = DataLoader(LazyDataset(dataset, transforms.Compose([feature_extractor])), batch_size=BATCH_SIZE, shuffle=False)
    result = []
    with torch.no_grad():
       for _, batch in tqdm(enumerate(dataloader)):
            frames = torch.squeeze(batch[0]['pixel_values'])
            predicted = model(frames)
            predicted = predicted.logits.argmax(dim=-1)
            result.extend(predicted.tolist())
    return result

def multi_process_prediction(model, feature_extractor, dataset: ImageFolder, processes=8):
    #split dataset
    base_length = len(dataset) // processes
    remaining_length = len(dataset) % processes
    lengths = [base_length] * processes
    lengths[-1] += remaining_length  # Add the remaining length to the last part
    datasets = random_split(dataset, lengths)

    futures = []
    with ProcessPoolExecutor(max_workers=processes) as executor:
        for ds in datasets:
            futures.append(executor.submit(single_process_prediction, model, feature_extractor, ds))
    
    
    result = []
    for future in futures:
        result.extend(future.result())
    return result

if __name__ == "__main__":
    model = load_model()
    feature_extractor = FeatureExtractor("microsoft/beit-base-patch16-224-pt22k-ft22k")
    dataset = ImageFolder("data/kaggle-ds")
    # take a sample of 200 images
    dataset = torch.utils.data.Subset(dataset, range(200))
    start_time = time.time()
    single_process_result = single_process_prediction(model, feature_extractor, dataset)
    single_process_time = time.time() - start_time
    print(f"Single process prediction time: {single_process_time:.2f} seconds")

    start_time = time.time()
    multi_process_result = multi_process_prediction(model, feature_extractor, dataset, processes=PROCESS_COUNT)
    multi_process_time = time.time() - start_time
    print(f"Multi-process prediction time: {multi_process_time:.2f} seconds")

    print(f"Speedup: {single_process_time / multi_process_time:.2f} times")
    
    with open("garbage-classifier/result.txt", "w") as f:
        f.write(f"Single process prediction time: {single_process_time:.2f} seconds\n")
        f.write(f"{PROCESS_COUNT}-process prediction time: {multi_process_time:.2f} seconds\n")
        f.write(f"Speedup: {single_process_time / multi_process_time:.2f} times\n")

