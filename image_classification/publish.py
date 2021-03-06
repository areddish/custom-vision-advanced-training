from config import config

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import *

# TODO: Fill in config.py with the correct values.
TRAINING_KEY = config["TrainingKey"]
PREDICTION_RESOURCE_ID = config["PredictionResourceId"]
SAMPLE_PROJECT_NAME = config["ProjectName"]
PUBLISH_ITERATION_NAME = config["PublishName"]
ENDPOINT = config["Endpoint"]

# Create an API client
trainer = CustomVisionTrainingClient(TRAINING_KEY, endpoint=ENDPOINT)

# Get our project we trained
project = next(project for project in trainer.get_projects() if project.name == SAMPLE_PROJECT_NAME)
if not project:
    print (f"Couldn't find project {SAMPLE_PROJECT_NAME}")
    exit(-1)

# Get the first trained iteration and publish that, change this if you want to publish a different iteration.
iteration = next(iter for iter in trainer.get_iterations(project.id) if iter.status == "Completed")
if not iteration:
    print (f"Couldn't find a trained iteration in {SAMPLE_PROJECT_NAME}")
    exit(-1)

# publish the project
print (f"Publishing {SAMPLE_PROJECT_NAME} as {PUBLISH_ITERATION_NAME}")
trainer.publish_iteration(project.id, iteration.id, PUBLISH_ITERATION_NAME, PREDICTION_RESOURCE_ID)