from pathlib import Path
from json import load
import os
import wandb
import torch

from garbage_data import GarbageData, extract_tar_gz
from model_utils import get_model, score_model
from trainer import get_optimizer, train_epoch
from model_card import create_model_card, save_model_card
from alibi_detect.cd.pytorch import preprocess_drift
from alibi_detect.utils.saving import save_detector
from alibi_detect.cd import KSDrift
from functools import partial
from alibi_detect.cd.pytorch import HiddenOutput


def train(config_path: Path, train_path, test_path, output_dir):
    # load config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_config = load(config_path.open())

    # wandb
    wandb.init(project="garbage-classifier")

    # data
    train_path = extract_tar_gz(Path(train_path))
    garbage_data = GarbageData(train_path, extract_tar_gz(
        Path(test_path)), json_config['batch_size'])
    train_dataloader = garbage_data.get_train_loader()
    val_dataloader = garbage_data.get_test_loader()

    # model
    classes = len(os.listdir(train_path))
    model = get_model(json_config['encoder'],
                      json_config['dropout'], json_config['fc_layer_size'], classes)
    model.to(device)

    # optimizer
    optimizer = get_optimizer(
        model, json_config['optimizer'], json_config['learning_rate'])

    # training loop
    f1_score = 0
    avg_loss = 0
    for epoch in range(json_config['epochs']):
        print(f'Epoch {epoch} started...')
        avg_loss = train_epoch(model, train_dataloader, optimizer, device)
        wandb.log({"loss": avg_loss})
        f1_score = score_model(model, val_dataloader, device)
        wandb.log({'val_f1': f1_score})

    os.makedirs(output_dir, exist_ok=True)
    # save model
    torch.save(model, os.path.join(output_dir, 'model.pth'))

    # save drift detector
    save_drift_detector(model, val_dataloader, output_dir, json_config['batch_size'])

    print(f"Model saved as {os.path.join(output_dir, 'model.pth')}")
    # create card
    _create_card(json_config['optimizer'], json_config['learning_rate'], json_config['batch_size'], json_config['epochs'],
                 json_config['fc_layer_size'], json_config['dropout'], f1_score, Path(output_dir) / 'card.md')


def save_drift_detector(model, dataloader, output_dir, batch_size):
    drift_detector = partial(preprocess_drift, model=HiddenOutput(model, layer=-1), batch_size=batch_size)
    X_ref = torch.cat([data[0]['pixel_values'] for data in dataloader], dim=0)
    X_ref = torch.squeeze(X_ref).numpy()
    cd = KSDrift(x_ref=X_ref, preprocess_fn=drift_detector, p_val=.05, alternative='two-sided')
    save_detector(cd, Path(output_dir) / 'drift_detector')

def _create_card(optimizer, learning_rate, batch_size, epochs, fc_layer_size, dropout, f1_score, path):
    model_name = "UWG Garbage Classifier"
    model_description = f"This UWG Garbage Classifier is an image classification model that distinguishes different types of garbage for special [UWG](https://nowaste.com.ua) stations. It is built on the top of the [BEiT](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k) model from the transformers library, which is used as encoder. The model is fine-tuned with a custom linear classifier. Fully connected layer size={fc_layer_size} and dropout rate={dropout} are used."
    data_details = "The dataset used for training and validation is [Kaggle Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification). This dataset has 15,150 images from 12 different classes of household garbage; paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, and trash."
    preprocessing_details = "Used augmentation technics: random affine transformations + horizontal flips. After that used BEiT encoder to produce embeddings for the dataset images."
    training_details = f"Trained using {optimizer} optimizer and cross entropy as loss. Learning rate: {learning_rate:.5f}, batch size: {batch_size}, number of epochs: {epochs}."
    evaluation_details = f"The performance of the model is evaluated using the macro F1 score. This score considers both precision and recall and computes their harmonic mean, which provides a balanced measure of the model's performance, taking into account both false positives and false negatives. Model managed to get F1 score ~ {f1_score:.3f} using 20% as validation set from provided dataset."
    usage = "To use this model you can use [train.py](https://github.com/rosklyar/projector-ml-ops/blob/main/garbage-classifier/garbage-classifier/train.py) to prepare the artifact and then [inference.py](https://github.com/rosklyar/projector-ml-ops/blob/main/garbage-classifier/garbage-classifier/inference.py) to make predictions."
    limitations = "The model's performance is dependent on the quality and quantity of the provided dataset. The model may not perform well on new types of garbage that were not present in the training dataset. Additionally, the pre-trained BeitForImageClassification model might not be the best choice for this specific task, and other architectures may yield better performance."

    model_card = create_model_card(
        model_name,
        model_description,
        data_details,
        preprocessing_details,
        training_details,
        evaluation_details,
        usage,
        limitations
    )

    save_model_card(path, model_card)
    print(f"Model Card saved as {path}")
