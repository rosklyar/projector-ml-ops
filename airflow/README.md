# How to run airflow DAGs

## Setup airflow on Windows using wsl2
1. Setup using this [guide](https://www.freecodecamp.org/news/install-apache-airflow-on-windows-without-docker/).
2. Create db `airflow db init`
3. Add admin user: `airflow users create --username admin --firstname FIRST_NAME --lastname LAST_NAME --role Admin --email admin@example.com`
4. Start scheduler and webserver: `airflow scheduler` and `airflow webserver`

## Kubernetes connect
1. Go to localhost:8080 and login with created admin user.
2. Copy your usr/.kube/config to your airflow folder and add path to it to 'kubernetes_default' connection in Admin->Connections tab.
3. Create volumes in your kubernetes cluster: `kubectl create -f .\airflow\volumes\airflow-volumes.yaml`

## Run the dags
1. Add needed environment variables for your Dags:
 - `airflow variables set s3_access_key S3_ACCESS_KEY`
 - `airflow variables set s3_secret_key S3_SECRET_KEY`
 - `airflow variables set wandb_api_key WANDB_API_KEY`
2. Add your dags to the $AIRFLOW_HOME/dags folder
3. Navigate to localhost:8080 and enable your dags
4. Run your dags with input parameters
