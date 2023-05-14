import wandb
from pathlib import Path

def upload_to_registry(model_name: str, model_path: Path):
    with wandb.init() as _:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(model_path / "config.json")
        art.add_file(model_path / "model.pth")
        art.add_file(model_path / "README.md")
        wandb.log_artifact(art)