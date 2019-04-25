import os
import config
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
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets", "test")

# Create an API client
trainer = CustomVisionTrainingClient(TRAINING_KEY, endpoint=ENDPOINT)

# Get our project we trained
project = None
print (f"Finding project: {SAMPLE_PROJECT_NAME}")
for p in trainer.get_projects():
    if p.name == SAMPLE_PROJECT_NAME:
        project = p
        break

if not project:
    print (f"Couldn't find project {SAMPLE_PROJECT_NAME}")
    exit(-1)

# Create a Prediction API client
predictor = CustomVisionPredictionClient(PREDICITON_KEY, endpoint=ENDPOINT)

# Use the file in test folder to do a prediction
with open(os.path.join(IMAGES_FOLDER, "test_od_image.jpg"), mode="rb") as test_data:
    results = predictor.detect_image(project.id, PUBLISH_ITERATION_NAME, test_data.read())

# Display the results.
for prediction in results.predictions:
    print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))


