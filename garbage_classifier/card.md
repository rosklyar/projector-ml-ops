# UWG Garbage Classifier

## Model Description

This UWG Garbage Classifier is an image classification model that distinguishes different types of garbage for special [UWG](https://nowaste.com.ua) stations. It is built on the top of the [BEiT](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k) model from the transformers library, which is used as encoder. The model is fine-tuned with a custom linear classifier. Fully connected layer size=128 and dropout rate=0.2 are used.

## Dataset

The dataset used for training and validation is [Kaggle Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification). This dataset has 15,150 images from 12 different classes of household garbage; paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, and trash.

## Training procedure

### Preprocessing
Used augmentation technics: random affine transformations + horizontal flips. After that used BEiT encoder to produce embeddings for the dataset images.

### Training
Trained using adam optimizer and cross entropy as loss. Learning rate: 0.00029, batch size: 8, number of epochs: 5.

#### Training System Hardware
- GPU: 1
- GPU type: NVIDIA GeForce RTX 3070 Laptop GPU
- CPU: 16

## Evaluation

The performance of the model is evaluated using the macro F1 score. This score considers both precision and recall and computes their harmonic mean, which provides a balanced measure of the model's performance, taking into account both false positives and false negatives. Model managed to get F1 score ~ 0.905 using 20% as validation set from provided dataset.

## Usage

To use this model you can use [train.py](https://github.com/rosklyar/projector-ml-ops/blob/main/garbage-classifier/garbage-classifier/train.py) to prepare the artifact and then [inference.py](https://github.com/rosklyar/projector-ml-ops/blob/main/garbage-classifier/garbage-classifier/inference.py) to make predictions.

## Limitations

The model's performance is dependent on the quality and quantity of the provided dataset. The model may not perform well on new types of garbage that were not present in the training dataset. Additionally, the pre-trained BeitForImageClassification model might not be the best choice for this specific task, and other architectures may yield better performance.
