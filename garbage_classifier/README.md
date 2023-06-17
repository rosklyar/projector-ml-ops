## Training routine

You can train model using next cli commands consecutively:
1. Load data from S3 bucket to local folder
```
python .\garbage_classifier\garbage_classifier\cli.py load-train-data S3_ACCESS_KEY S3_SECRET_KEY BUCKET_NAME SUBFOLDER C:\Users\rskliar\PycharmProjects\projector-ml-ops\tmp\
```
2. Train model
```
python .\garbage_classifier\garbage_classifier\cli.py train .\garbage_classifier\garbage_classifier\data\config.json .\tmp\train.tar.gz .\tmp\test.tar.gz C:\Users\rskliar\PycharmProjects\projector-ml-ops\tmp\model\
```
3. Upload model to wandb registry
```
python .\garbage_classifier\garbage_classifier\cli.py upload-to-registry uwg-classifier .\tmp\model\ .\tmp\input\config.json
```
## Inference routine

You can make predictions using next cli commands consecutively:
1. Download model from wandb registry
```
python .\garbage_classifier\garbage_classifier\cli.py download-from-registry uwg-classifier v0
```
2. Download data from S3 bucket to local folder
```
python .\garbage_classifier\garbage_classifier\cli.py load-data S3_ACCESS_KEY S3_SECRET_KEY BUCKET_NAME SUBFOLDER C:\Users\rskliar\PycharmProjects\projector-ml-ops\tmp\data\
```
3. Make predictions
```
python .\garbage_classifier\garbage_classifier\cli.py make-inference .\artifacts\uwg-classifier-v0 .\tmp\data\data.tar.gz
```
