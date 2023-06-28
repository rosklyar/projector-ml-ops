from transformers import BeitForImageClassification
from sklearn.metrics import f1_score as measure_f1_score
from tqdm import tqdm
import torch
from torch import nn

class BeitImageClassificationModel(nn.Module):
    
    def __init__(self, model_name, dropout_rate, fc_layer_size, n_classes):
        super().__init__()
        self.model = BeitForImageClassification.from_pretrained(model_name)
        self.model.config.num_labels = n_classes
        self.model.classifier = nn.Sequential(
            nn.Linear(768, fc_layer_size),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_layer_size, n_classes)
        )
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.fc_layer_size = fc_layer_size
        self.n_classes = n_classes

    def forward(self, x):
        return self.model(x)

def get_model(model_name, dropout_rate, fc_layer_size, n_classes):
    return BeitImageClassificationModel(model_name, dropout_rate, fc_layer_size, n_classes)


@torch.no_grad()
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
    for _, batch in tqdm(enumerate(dataloader)):
        frames = torch.squeeze(batch[0]['pixel_values']).to(device)
        labels = batch[1].to(device)
        predicted = model(frames)
        predicted = predicted.logits.argmax(dim=-1)
        result.extend(predicted.cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())
    f1_score = measure_f1_score(targets, result, average='macro')
    return f1_score
