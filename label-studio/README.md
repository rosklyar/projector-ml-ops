# Instuction to deploy on Kubernetes
1. Run Label Studio with `kubectl apply -f .\label-studio\label-studio-pod.yaml` on kubernetes
2. Port forward to Label Studio UI with `kubectl port-forward pod/label-studio 8080`