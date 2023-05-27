# How to run Kubeflow pipelines

## Setup Kubeflow pipelines using kubectl
1. Create resources and apply them using kubectl
```
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=1.8.5" > kubeflow/res.yaml
kubectl create -f kubeflow/res.yaml
kubectl kustomize "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=1.8.5" > kubeflow/pipelines.yaml
kubectl create -f kubeflow/pipelines.yaml
```
2. Portforward Kubeflow UI
```
kubectl port-forward --address=0.0.0.0 svc/ml-pipeline-ui 8888:80 -n kubeflow
```

## Run your pipelines
1. Add pipelines 
```
python .\garbage_classifier\pipelines\kubeflow-pipelines.py http://localhost:8888
```
2. Run created pipelines from UI http://localhost:8888
