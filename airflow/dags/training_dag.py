from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)
from kubernetes.client import models as k8s
from airflow.models import Variable
from airflow.models.param import Param

volume = k8s.V1Volume(
    name="training-storage",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="training-storage"
    ),
)
volume_mount = k8s.V1VolumeMount(
    name="training-storage", mount_path="/var/tmp/", sub_path=None
)
IMAGE = "rostyslavskliar/garbage-classifier-trainer:latest"

with DAG(
    start_date=datetime(2023, 1, 1),
    params={
        "bucket": Param("nowastecomua", type="string"),
        "folder": Param("test", type="string"),
    },
    catchup=False,
    schedule_interval=None,
    dag_id="training_dag",
) as dag:
    s3_access_key = Variable.get("s3_access_key")
    s3_secret_key = Variable.get("s3_secret_key")
    wandb_api_key = Variable.get("wandb_api_key")

    clean_storage_before_start = KubernetesPodOperator(
        name="clean_storage_before_start",
        image=IMAGE,
        cmds=["rm", "-rf", "/var/tmp/"],
        task_id="clean_storage_before_start",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    load_data = KubernetesPodOperator(
        name="load_data",
        image=IMAGE,
        cmds=[
            "python",
            "garbage_classifier/cli.py",
            "load-train-data",
            s3_access_key,
            s3_secret_key,
            dag.params["bucket"],
            dag.params["folder"],
            "/var/tmp/data/",
        ],
        task_id="load_data",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    train_model = KubernetesPodOperator(
        name="train_model",
        image=IMAGE,
        cmds=[
            "python",
            "garbage_classifier/cli.py",
            "train",
            "garbage_classifier/data/config.json",
            "/var/tmp/data/train.tar.gz",
            "/var/tmp/data/test.tar.gz",
            "/var/tmp/model/",
        ],
        task_id="train_model",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    upload_model = KubernetesPodOperator(
        name="upload_model",
        image=IMAGE,
        cmds=[
            "python",
            "garbage_classifier/cli.py",
            "upload-to-registry",
            "garbage-classifier-model",
            "/var/tmp/model/",
            "/var/tmp/data/input/config.json",
        ],
        task_id="upload_model",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    clean_up = KubernetesPodOperator(
        name="clean_up",
        image=IMAGE,
        cmds=["rm", "-rf", "/var/tmp/"],
        task_id="clean_up",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
        trigger_rule="all_done",
    )

    clean_storage_before_start >> load_data >> train_model >> upload_model >> clean_up
