import os
import uuid
from typing import Optional

import kfp
import typer
from kfp import dsl
from kubernetes.client.models import V1EnvVar

IMAGE = "rostyslavskliar/garbage-classifier-training:latest"


@dsl.pipeline(name="garbage_classifier_traininig_pipeline", description="Pipeline for training garbage classifier from scratch")
def nlp_traininig_pipeline(s3_access_key, s3_secret_key, s3_bucket, s3_prefix, wandb_api_key):

    load_data = dsl.ContainerOp(
        name="load_data",
        command=["python", "garbage_classifier/cli.py", "load-data",
                 s3_access_key, s3_secret_key, s3_bucket, s3_prefix],
        image=IMAGE,
        file_outputs={"train": "/data/real-ds/train.tar.gz",
                      "test": "/data/real-ds/test.tar.gz"},
    )
    load_data.execution_options.caching_strategy.max_cache_staleness = "P0D"

    train_model = dsl.ContainerOp(
        name="train",
        command=["python", "garbage_classifier/cli.py",
                 "train" "/data/config.json"],
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(
                load_data.outputs["train"], path="/data/real-ds/train.tar.gz"),
            dsl.InputArgumentPath(
                load_data.outputs["test"], path="/data/real-ds/test.tar.gz")
        ],
        file_outputs={
            "config": "/data/config.json",
            "model": "/model/model.pth",
            "model_card": "/data/real-ds/README.md",
        },
    )

    upload_model = dsl.ContainerOp(
        name="upload_model ",
        command=["python", "garbage_classifier/cli.py",
                 "upload-to-registry", "garbage-classifier-kf", "/tmp/results"],
        image=IMAGE,
        artifact_argument_paths=[
            dsl.InputArgumentPath(
                train_model.outputs["config"], path="/tmp/results/config.json"),
            dsl.InputArgumentPath(
                train_model.outputs["model"], path="/tmp/results/model.pth"),
            dsl.InputArgumentPath(
                train_model.outputs["model_card"], path="/tmp/results/README.md"),
        ],
    )

    env_var_project = V1EnvVar(name="WANDB_PROJECT", value="garbage-classifier")
    upload_model = upload_model.add_env_variable(env_var_project)

    env_var_password = V1EnvVar(name="WANDB_API_KEY", value=wandb_api_key)
    upload_model = upload_model.add_env_variable(env_var_password)


def compile_pipeline() -> str:
    path = ".\garbage_classifier\pipelines\garbage_classifier_traininig_pipeline.yaml"
    kfp.compiler.Compiler().compile(nlp_traininig_pipeline, path)
    return path


def create_pipeline(client: kfp.Client, namespace: str):
    print("Creating experiment...")
    _ = client.create_experiment("training", namespace=namespace)

    print("Uploading pipeline...")
    name = "garbage-classifier-training"
    if client.get_pipeline_id(name) is not None:
        print("Pipeline exists - upload new version...")
        pipeline_prev_version = client.get_pipeline(
            client.get_pipeline_id(name))
        version_name = f"{name}-{uuid.uuid4()}"
        pipeline = client.upload_pipeline_version(
            pipeline_package_path=compile_pipeline(),
            pipeline_version_name=version_name,
            pipeline_id=pipeline_prev_version.id,
        )
    else:
        pipeline = client.upload_pipeline(
            pipeline_package_path=compile_pipeline(), pipeline_name=name)
    print(f"Pipeline {pipeline.id}")


def auto_create_pipelines(
    host: str,
    namespace: Optional[str] = None,
):
    client = kfp.Client(host=host)
    create_pipeline(client=client, namespace=namespace)


if __name__ == "__main__":
    typer.run(auto_create_pipelines)
