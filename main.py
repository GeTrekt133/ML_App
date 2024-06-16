from model import model_pipeline
from model import DenseNet201
from fastapi import FastAPI, UploadFile
from PIL import Image
import torch
import io


app = FastAPI()
model = DenseNet201()
model.load_state_dict(torch.load('model_weights.ckpt', map_location=torch.device('cpu')))


@app.post("/answer")
def ask(image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    result = model_pipeline(image, model)
    return result
