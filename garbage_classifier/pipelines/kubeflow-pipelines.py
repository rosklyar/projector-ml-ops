import uuid
from typing import Optional

import kfp
import typer
from kfp import dsl
from kubernetes.client.models import V1EnvVar

IMAGE = "rostyslavskliar/garbage-classifier-trainer:latest"


@dsl.pipeline(name="garbage_classifier_traininig_pipeline", description="Pipeline for training garbage classifier from scratch")
def garbage_classifier_traininig_pipeline(s3_access_key, s3_secret_key, s3_bucket, s3_prefix, wandb_api_key):

    load_data = dsl.ContainerOp(
        name="load_data",
        command=["python", "garbage_classifier/cli.py", "load-train-data",
                 s3_access_key, s3_secret_key, s3_bucket, s3_prefix, "/tmp/data/"],
        image=IMAGE,
        file_outputs={"train": "/tmp/data/train.tar.gz",
                      "test": "/tmp/data/test.tar.gz",
                      "classes": "/tmp/data/input/config.json"},
    )
    load_data.execution_options.caching_strategy.max_cache_staleness = "P0D"

    train_model = dsl.ContainerOp(
        name="train",
        command=["python", "garbage_classifier/cli.py",
                 "train", "garbage_classifier/data/config.json", "/tmp/data/train.tar.gz", "/tmp/data/test.tar.gz", "/tmp/model/"],
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(
                load_data.outputs["train"], path="/tmp/data/train.tar.gz"),
            dsl.InputArgumentPath(
                load_data.outputs["test"], path="/tmp/data/test.tar.gz")
        ],
        file_outputs={
            "model": "/tmp/model/model.pth",
            "model_card": "/tmp/model/card.md",
        },
    )

    upload_model = dsl.ContainerOp(
        name="upload_model",
        command=["python", "garbage_classifier/cli.py",
                 "upload-to-registry", "uwg-classifier", "/tmp/model/", "/tmp/model/config.json"],
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(
                load_data.outputs["classes"], path="/tmp/model/config.json"),
            dsl.InputArgumentPath(
                train_model.outputs["model"], path="/tmp/model/model.pth"),
            dsl.InputArgumentPath(
                train_model.outputs["model_card"], path="/tmp/model/card.md"),
        ],
    )

    env_var_project = V1EnvVar(
        name="WANDB_PROJECT", value="garbage-classifier")
    train_model = train_model.add_env_variable(env_var_project)
    upload_model = upload_model.add_env_variable(env_var_project)

    env_var_password = V1EnvVar(name="WANDB_API_KEY", value=wandb_api_key)
    train_model = train_model.add_env_variable(env_var_password)
    upload_model = upload_model.add_env_variable(env_var_password)


@dsl.pipeline(name="garbage_classifier_inference_pipeline", description="Pipeline for inference data with garbage classifier")
def garbage_classifier_inference_pipeline(s3_access_key, s3_secret_key, s3_bucket, s3_prefix, wandb_api_key, model_name: str = 'uwg-classifier', model_version: str = 'v0'):

    load_data = dsl.ContainerOp(
        name="load_data",
        command=["python", "garbage_classifier/cli.py", "load-data",
                 s3_access_key, s3_secret_key, s3_bucket, s3_prefix, "/tmp/data/"],
        image=IMAGE,
        file_outputs={"data": "/tmp/data/data.tar.gz"},
    )
    load_data.execution_options.caching_strategy.max_cache_staleness = "P0D"

    download_model = dsl.ContainerOp(
        name="download_model",
        command=["python", "garbage_classifier/cli.py",
                 "download-from-registry", model_name, model_version],
        image=IMAGE,
        file_outputs={
            "model": f'/artifacts/{model_name.value}-{model_version.value}/model.pth'
        },
    )

    env_var_project = V1EnvVar(
        name="WANDB_PROJECT", value="garbage-classifier")
    download_model = download_model.add_env_variable(env_var_project)

    env_var_password = V1EnvVar(name="WANDB_API_KEY", value=wandb_api_key)
    download_model = download_model.add_env_variable(env_var_password)

    inference = dsl.ContainerOp(
        name="inference",
        command=["python", "garbage_classifier/cli.py",
                 "make-inference", f"/tmp/model/model.pth", "/tmp/data/data.tar.gz"],
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(
                load_data.outputs["data"], path="/tmp/data/data.tar.gz"),
            dsl.InputArgumentPath(
                download_model.outputs["model"], path="/tmp/model/model.pth")
        ],
        file_outputs={
            "result": "/tmp/result.csv"
        },
    )


def compile_pipeline(name: str, function) -> str:
    path = f".\garbage_classifier\pipelines\{name}-pipeline.yaml"
    kfp.compiler.Compiler().compile(function, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str, name: str, function):
    print("Creating experiment...")
    _ = client.create_experiment("training", namespace=namespace)

    print("Uploading pipeline...")
    if client.get_pipeline_id(name) is not None:
        print("Pipeline exists - upload new version...")
        pipeline_prev_version = client.get_pipeline(
            client.get_pipeline_id(name))
        version_name = f"{name}-{uuid.uuid4()}"
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=compile_pipeline(name, function),
            pipeline_version_name=version_name,
            pipeline_id=pipeline_prev_version.id,
        )
    else:
        pipeline = client.upload_pipeline(
            pipeline_package_path=compile_pipeline(name, function), pipeline_name=name)
    print(f"Pipeline {pipeline.id}")


def auto_create_pipelines(
    host: str,
    namespace: Optional[str] = None,
):
    client = kfp.Client(host=host)
    create_pipeline(client=client, namespace=namespace, name="garbage-classifier-training",
                    function=garbage_classifier_traininig_pipeline)
    create_pipeline(client=client, namespace=namespace, name="garbage-classifier-inference",
                    function=garbage_classifier_inference_pipeline)


if __name__ == "__main__":
    typer.run(auto_create_pipelines)
