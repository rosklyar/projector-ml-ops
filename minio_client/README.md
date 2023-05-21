# Instuction to deploy on Kubernetes
1. Run minio with `kubectl apply -f ./minio/pod-minio.yaml` on kubernetes
2. Port forward to minio service with `kubectl port-forward pod/minio 9000 9090`
