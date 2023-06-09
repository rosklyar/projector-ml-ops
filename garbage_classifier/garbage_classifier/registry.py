import wandb
from pathlib import Path

def upload_to_registry(model_name: str, model_path: Path, classes_json: Path):
    with wandb.init(project="garbage-classifier", entity="rosklyar") as _:
        art = wandb.Artifact(model_name, type="model")
        art.add_file(classes_json)
        art.add_file(model_path / "model.pth")
        art.add_file(model_path / "card.md")
        art.add_dir(model_path / "drift_detector", name="drift_detector")
        wandb.log_artifact(art)

def download_from_registry(artifact_name, artifact_version):
    with wandb.init(project="garbage-classifier", entity="rosklyar") as run:
        artifact = run.use_artifact(f'rosklyar/garbage-classifier/{artifact_name}:{artifact_version}')
        artifact_dir = artifact.download()
        run.finish()
        return artifact_dir