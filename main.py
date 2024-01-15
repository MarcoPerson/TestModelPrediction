from typing import Annotated

from fastapi import FastAPI, Query
from PIL import Image
app = FastAPI()

from ImageClassifierPredictor import ImageClassifier

# Specify the path to the trained model
model_path = 'model/inceptionv3_multi_label_model_All.pth'

# Create an instance of the ImageClassifier
classifier = None
    
@app.get("/")
def read_root():
    return {"Message": "Bienvenue sur le microservice de gestion du Model !"}


@app.get("/predict_one_image")
def predict(image_url: str):
    prediction = classifier.predict_one(image_url)
    return prediction

@app.get("/predict_many_image")
def predict_many_image(list_of_images_url: Annotated[list[str], Query()]):
    predictions = classifier.predict_multiple(list_of_images_url)
    return predictions