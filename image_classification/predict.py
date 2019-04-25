import os
from config import config
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import *
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# TODO: Fill in config.py with the correct values.
TRAINING_KEY = config["TrainingKey"]
PREDICITON_KEY = config["PredictionKey"]
SAMPLE_PROJECT_NAME = config["ProjectName"]
PUBLISH_ITERATION_NAME = config["PublishName"]
ENDPOINT = config["Endpoint"]

# Image folder with data sets
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset", "test")

# Create an API client
trainer = CustomVisionTrainingClient(TRAINING_KEY, endpoint=ENDPOINT)

# Get our project we trained
project = next(project for project in trainer.get_projects() if project.name == SAMPLE_PROJECT_NAME)
if not project:
    print (f"Couldn't find project {SAMPLE_PROJECT_NAME}")
    exit(-1)

# Create a Prediction API client
predictor = CustomVisionPredictionClient(PREDICITON_KEY, endpoint=ENDPOINT)

# Use the file in test folder to do a prediction
with open(os.path.join(IMAGES_FOLDER, "test_image.jpg"), mode="rb") as test_data:
    results = predictor.classify_image(project.id, PUBLISH_ITERATION_NAME, test_data.read())

    # Display the results.
    for prediction in results.predictions:
        print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))