from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam


def get_optimizer(model, optimizer, lr):
    if optimizer == "sgd":
        optimizer = SGD(model.parameters(),
                        lr=lr, momentum=0.9)
    elif optimizer == "adam":
        optimizer = Adam(model.parameters(),
                         lr=lr)
    return optimizer


def train_epoch(model, dataloader, optimizer, device='cpu'):
    cumu_loss = 0
    for _, batch in tqdm(enumerate(dataloader)):
        frames, labels = torch.squeeze(batch[0]['pixel_values']).to(
            device), batch[1].to(device)
        optimizer.zero_grad()

        # forward pass
        loss = F.cross_entropy(model(frames).logits, labels)
        cumu_loss += loss.item()

        # backward pass
        loss.backward()
        optimizer.step()

    return cumu_loss / len(dataloader)