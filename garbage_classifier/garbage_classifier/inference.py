from concurrent.futures import ProcessPoolExecutor
from torchvision import torch, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pathlib import Path
from alibi_detect.utils.saving import load_detector
from timeit import default_timer as timer

from tqdm import tqdm
import time

from garbage_data import LazyDataset, FeatureExtractor, extract_tar_gz

BATCH_SIZE = 16
PROCESS_COUNT = 8
DRIFT_LABELS = ['No!', 'Yes!']

def load_model(model_path, device='cpu'):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def predict(model, batch):
    frames = torch.squeeze(batch[0]['pixel_values'])
    predicted = model(frames)
    return predicted

def single_process_prediction(model, feature_extractor, dataset: ImageFolder, device='cpu'):
    dataloader = DataLoader(LazyDataset(dataset, transforms.Compose([feature_extractor])), batch_size=BATCH_SIZE, shuffle=False)
    result = []
    with torch.no_grad():
       for _, batch in tqdm(enumerate(dataloader)):
            frames = torch.squeeze(batch[0]['pixel_values']).to(device)
            predicted = model(frames)
            predicted = predicted.logits.argmax(dim=-1)
            result.extend(predicted.tolist())
    return result

def multi_process_prediction(model, feature_extractor, dataset: ImageFolder, processes=8, device='cpu'):
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

def make_inference(model_path: Path, data_archive_path):
    data_path = extract_tar_gz(data_archive_path)
    dataset = ImageFolder(data_path)
    feature_extractor = FeatureExtractor("microsoft/beit-base-patch16-224-pt22k-ft22k")
    
    # load drift detector
    cd = load_detector(model_path / "drift_detector")
    make_drift_predictions(cd, createInputsForDriftDetector(dataset, feature_extractor))
    
    # inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path / "model.pth", device)
    paths = [path for path, _ in dataset.imgs]
    multi_process_result = single_process_prediction(model, feature_extractor, dataset, device)
    with open("result.csv", "w") as f:
        for path, label in zip(paths, multi_process_result):
            f.write(f"{path},{label}\n")

def make_drift_predictions(cd, x_h0):
    t = timer()
    preds = cd.predict(x_h0)
    dt = timer() - t
    print('No corruption')
    print('Drift? {}'.format(DRIFT_LABELS[preds['data']['is_drift']]))
    print('Feature-wise p-values:')
    print(preds['data']['p_val'])
    print(f'Time (s) {dt:.3f}')
    if preds['data']['is_drift']:
        raise Exception("Drift detected")

def createInputsForDriftDetector(dataset: ImageFolder, feature_extractor: FeatureExtractor):
    dataloader = DataLoader(LazyDataset(dataset, transforms.Compose([feature_extractor])), batch_size=BATCH_SIZE, shuffle=False)
    X_ref = torch.cat([data[0]['pixel_values'] for data in dataloader], dim=0)
    X_ref = torch.squeeze(X_ref).numpy()
    return X_ref

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("model/model.pth", device)
    feature_extractor = FeatureExtractor("microsoft/beit-base-patch16-224-pt22k-ft22k")
    dataset = ImageFolder("data/kaggle-ds")
    # take a sample of 200 images
    dataset = torch.utils.data.Subset(dataset, range(200))
    start_time = time.time()
    single_process_result = single_process_prediction(model, feature_extractor, dataset, device)
    single_process_time = time.time() - start_time
    print(f"Single process prediction time: {single_process_time:.2f} seconds")

    start_time = time.time()
    multi_process_result = multi_process_prediction(model, feature_extractor, dataset, processes=PROCESS_COUNT, device=device)
    multi_process_time = time.time() - start_time
    print(f"Multi-process prediction time: {multi_process_time:.2f} seconds")

    print(f"Speedup: {single_process_time / multi_process_time:.2f} times")
    
    with open("garbage-classifier/result.txt", "w") as f:
        f.write(f"Single process prediction time: {single_process_time:.2f} seconds\n")
        f.write(f"{PROCESS_COUNT}-process prediction time: {multi_process_time:.2f} seconds\n")
        f.write(f"Speedup: {single_process_time / multi_process_time:.2f} times\n")

