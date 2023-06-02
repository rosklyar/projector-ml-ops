import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from serving.predictor import Predictor

app = FastAPI()

predictor = Predictor.default_from_model_registry(os.getenv("MODEL_ID"), os.getenv("MODEL_PATH"))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )

@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    try:
        image = Image.open(image_file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to open image file")
    return {"result": predictor.predict(image)}

@app.get("/config")
async def config():
    return predictor.get_classes_config()
