## Triton deployment
### Docker run
1. Run docker image with mounted model_repository folder
```docker run --gpus=1 --rm --net=host -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models```

### k8s deployment
1. Create k8s deployment
```kubectl apply -f /web-app/k8s/triton.yaml```