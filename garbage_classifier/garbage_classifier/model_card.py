import torch
import psutil


def create_model_card(model_name, model_description, data_details, preprocessing_details, training_details, evaluation_details, usage, limitations):
    model_card = f"""# {model_name}

## Model Description

{model_description}

## Dataset

{data_details}

## Training procedure

### Preprocessing
{preprocessing_details}

### Training
{training_details}

#### Training System Hardware
- GPU: {torch.cuda.device_count()}
- GPU type: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
- CPU: {psutil.cpu_count()}

## Evaluation

{evaluation_details}

## Usage

{usage}

## Limitations

{limitations}
"""

    return model_card


def save_model_card(filename, model_card_content):
    with open(filename, 'w') as f:
        f.write(model_card_content)
        print(f"Model Card saved as {filename}")
