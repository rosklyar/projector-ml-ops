from datetime import datetime

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)
from kubernetes.client import models as k8s
from airflow.models import Variable
from airflow.models.param import Param

volume = k8s.V1Volume(
    name="inference-storage",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name="inference-storage"
    ),
)
volume_mount = k8s.V1VolumeMount(
    name="inference-storage", mount_path="/var/tmp/", sub_path=None
)
IMAGE = "rostyslavskliar/garbage-classifier-trainer:latest"

with DAG(
    start_date=datetime(2023, 1, 1),
    params={
        "bucket": Param("nowastecomua", type="string"),
        "folder": Param("test", type="string"),
        "model_name": Param("garbage-classifier-model", type="string"),
        "version": Param("v0", type="string"),
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
            "load-data",
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

    model_name = dag.params["model_name"]
    model_version = dag.params["version"]

    download_model = KubernetesPodOperator(
        name="download_model",
        image=IMAGE,
        cmds=[
            "python",
            "garbage_classifier/cli.py",
            "download-from-registry",
            model_name,
            model_version,
        ],
        task_id="download_model",
        in_cluster=False,
        namespace="default",
        volumes=[volume],
        volume_mounts=[volume_mount],
    )

    inference = KubernetesPodOperator(
        name="inference",
        image=IMAGE,
        cmds=[
            "python",
            "garbage_classifier/cli.py",
            "make-inference",
            f"/artifacts/{model_name}-{model_version}",
            "/var/tmp/data/data.tar.gz",
        ],
        task_id="inference",
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

    clean_storage_before_start >> load_data
    clean_storage_before_start >> download_model

    load_data >> inference
    download_model >> inference
    inference >> clean_up
