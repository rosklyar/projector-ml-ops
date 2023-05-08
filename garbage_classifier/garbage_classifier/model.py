from transformers import BeitForImageClassification
from sklearn.metrics import f1_score as measure_f1_score
from tqdm import tqdm
import torch

def get_model(dropout_rate, fc_layer_size, n_classes):
    model = BeitForImageClassification.from_pretrained(
        'microsoft/beit-base-patch16-224-pt22k-ft22k')

    layers = [torch.nn.Linear(768, fc_layer_size), torch.nn.Dropout(
        dropout_rate), torch.nn.Linear(fc_layer_size, n_classes)]

    model.classifier = torch.nn.Sequential(*layers)

    model.config.num_labels = n_classes
    
    return model

def score_model(model, dataloader, device='cpu'):
    """
    :param model:
    :param dataloader:
    :return:
        res: f1_score
    """
    print('Model scoring was started...')
    model.eval()
    dataloader.dataset.mode = 'eval'
    result = []
    targets = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(dataloader)):
            frames = torch.squeeze(batch[0]['pixel_values']).to(device)
            labels = batch[1].to(device)
            predicted = model(frames)
            predicted = predicted.logits.argmax(dim=-1)
            result.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    f1_score = measure_f1_score(targets, result, average='macro')
    return f1_score