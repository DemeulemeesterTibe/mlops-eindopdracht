import numpy as np
from tensorflow.keras.models import load_model  
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

SENTIMENTS = []
# inference\emotions-classification\INPUT_model_path\emotions\emotions.h5

model_path = os.path.join("emotions-classification","INPUT_model_path", "emotions", "emotions")
# print current working directory
print("Current working directory: {0}".format(os.getcwd()))
print("model_path",model_path)
print("Loading model...")
model = load_model(model_path)

@app.post('/get/sentiment')
async def uploadImage(text: str):


    predictions = model.predict()
    classifications = predictions.argmax(axis=1)

    return SENTIMENTS[classifications.tolist()[0]]

@app.get('/healthcheck')
def healthcheck():
    return {'status': 'Healty'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)