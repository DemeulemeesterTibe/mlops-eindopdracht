import numpy as np
from tensorflow.keras.models import load_model  
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
from utils import *

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
model_folder = os.path.join("emotions-classification","INPUT_model_path", "emotions")

SENTIMENTS = np.load(os.path.join(model_folder, "emotions-labels.npy"),allow_pickle=True)

model_path = os.path.join(model_folder, "emotions.h5")
# print current working directory
print("Loading model...")
model = load_model(model_path)

@app.post('/get/sentiment')
async def uploadImage(text: str):
    proccessed_text = preprocess(text)
    predictions = model.predict(proccessed_text)
    classifications = predictions.argmax(axis=1)

    return {'sentiment': SENTIMENTS[classifications.tolist()[0]]}

@app.get('/healthcheck')
def healthcheck():
    return {'status': 'Healty'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)