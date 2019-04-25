import os
from config import config

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import *

# TODO: Fill in config.py with the correct values.
TRAINING_KEY = config["TrainingKey"]
PREDICTION_RESOURCE_ID = config["PredictionResourceId"]
NOTIFY_EMAIL = config["NotifyEmail"]
SAMPLE_PROJECT_NAME = config["ProjectName"]
PUBLISH_ITERATION_NAME = config["PublishName"]
ENDPOINT = config["Endpoint"]
BUDGET_IN_HOURS = config["Budget"]

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset", "train")

# Create an API client
trainer = CustomVisionTrainingClient(TRAINING_KEY, endpoint=ENDPOINT)

# Create a new project
print (f"Creating project: {SAMPLE_PROJECT_NAME}")
project = trainer.create_project(SAMPLE_PROJECT_NAME)

# Upload images
# This walks a directory, using the folder name as the tag name. For each folder we create a tag and then upload all
# files in that directory associating them with that tag.
tags = os.listdir(IMAGES_FOLDER)
print (f"Uploading images from {IMAGES_FOLDER}")
for tag in tags:
    # Create the tag, and grab the id
    tag_id = trainer.create_tag(project.id, tag).id
    tag_dir = os.path.join(IMAGES_FOLDER, tag)
    for image in os.listdir(tag_dir):
        print (f"Uploading: {image} with tag {tag}")
        with open(os.path.join(tag_dir, image), mode="rb") as img_data: 
            trainer.create_images_from_data(project.id, img_data.read(), [ tag_id ])

# Advanced training
# Advanced training is triggered by passing the advanced training type and specifying a budget.
# You can optionally specify a notification email address to receive and email when training completes.
print ("Starting Training...")
trainer.train_project(project.id, training_type=TrainingType.advanced, reserved_budget_in_hours=BUDGET_IN_HOURS, notification_email_address=NOTIFY_EMAIL)