from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/inference/")
async def check(file: UploadFile):
    return {"bin": 0}