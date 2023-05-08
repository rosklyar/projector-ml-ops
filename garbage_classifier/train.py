import random
from os.path import join
import numpy as np
import torch
import wandb

from garbage_classifier.garbage_data import GarbageData
from garbage_classifier.config import config as opt
from garbage_classifier.model_card import create_model_card, save_model_card
from garbage_classifier.model import get_model, score_model
from garbage_classifier.trainer import get_optimizer, train_epoch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def train_model(config=None):
    with wandb.init(config=config):
        # config
        config = wandb.config

        # data
        garbage_data = GarbageData(opt.dataset_dir, config.batch_size)
        train_dataloader = garbage_data.get_train_loader()
        val_dataloader = garbage_data.get_val_loader()

        # model
        model = get_model(config.dropout, config.fc_layer_size, opt.classes)
        model.to(opt.device)

        # optimizer
        optimizer = get_optimizer(model, config.optimizer, config.lr)
        f1_score = 0

        for epoch in range(config.epochs):
            print(f'Epoch {epoch} started...')
            avg_loss = train_epoch(model, train_dataloader, optimizer, opt.device)
            wandb.log({"loss": avg_loss})
            f1_score = score_model(model, val_dataloader, opt.device)
            wandb.log({'val_f1': f1_score})
            torch.save(model, join(opt.checkpoint_dir,
                       f'epoch{epoch}_f1={round(f1_score, 5)}.pth'))
        # create card
        create_card(config, f1_score)
        # save model
        torch.save(model, join(opt.checkpoint_dir, 'weights', 'latest', 'model.pth'))
        art = wandb.Artifact(f'uwg-garbage-classifier-{wandb.run.id}', type="model")
        art.add_file(join(opt.checkpoint_dir, 'weights', 'latest', 'model.pth'), "model.pth")
        art.add_file("garbage_classifier/README.md", "README.md")
        wandb.log({"loss": avg_loss, "f1_score": f1_score})
        wandb.log_artifact(art)


def get_sweep_config():
    return {
        'method': 'random',
        'metric': {'name': 'val_f1', 'goal': 'maximize'},
        'parameters':  {
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            'fc_layer_size': {
                'values': [128, 256, 512]
            },
            'dropout': {
                'values': [0.1, 0.2]
            },
            'epochs': {'value': 5},
            'lr': {
                # a flat distribution between 0 and 0.1
                'distribution': 'uniform',
                'min': 0.0001,
                'max': 0.001
            },
            'batch_size': {
                # integers between 8 and 16
                # with evenly-distributed logarithms
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': 8,
                'max': 16,
            }
        }
    }


def create_card(config, f1_score):
    model_name = "UWG Garbage Classifier"
    model_description = f"This UWG Garbage Classifier is an image classification model that distinguishes different types of garbage for special [UWG](https://nowaste.com.ua) stations. It is built on the top of the [BEiT](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k) model from the transformers library, which is used as encoder. The model is fine-tuned with a custom linear classifier. Fully connected layer size={config.fc_layer_size} and dropout rate={config.dropout} are used."
    data_details = "The dataset used for training and validation is [Kaggle Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification). This dataset has 15,150 images from 12 different classes of household garbage; paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, and trash."
    preprocessing_details = "Used augmentation technics: random affine transformations + horizontal flips. After that used BEiT encoder to produce embeddings for the dataset images."
    training_details = f"Trained using {config.optimizer} optimizer and cross entropy as loss. Learning rate: {config.lr:.5f}, batch size: {config.batch_size}, number of epochs: {config.epochs}."
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

    filename = "garbage-classifier/README.md"
    save_model_card(filename, model_card)


if __name__ == '__main__':
    sweep_id = wandb.sweep(get_sweep_config(), project="garbage-classifier")
    wandb.agent(sweep_id, train_model, count=5)
