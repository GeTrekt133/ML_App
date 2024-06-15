from model import model_pipeline
from fastapi import FastAPI, UploadFile
from PIL import Image
import io


app = FastAPI()


@app.post("/")
def ask(image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    result = model_pipeline(image)
    return result

