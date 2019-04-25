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

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset", "train")

# Create an API client
trainer = CustomVisionTrainingClient(TRAINING_KEY, endpoint=ENDPOINT)

# Find the object detection domain
obj_detection_domain = next(domain for domain in trainer.get_domains() if domain.type == "ObjectDetection" and domain.name == "General")

# Create a new project
print (f"Creating project: {SAMPLE_PROJECT_NAME}")
project = trainer.create_project(SAMPLE_PROJECT_NAME, domain_id=obj_detection_domain.id)

# Upload images

# These bounding boxes are hard coded for this sample and could also be read from a file or some other source.
image_regions = {
    "fork_1.jpg": [ 0.145833328, 0.3509314, 0.5894608, 0.238562092 ],
    "fork_2.jpg": [ 0.294117659, 0.216944471, 0.534313738, 0.5980392 ],
    "fork_3.jpg": [ 0.09191177, 0.0682516545, 0.757352948, 0.6143791 ],
    "fork_4.jpg": [ 0.254901975, 0.185898721, 0.5232843, 0.594771266 ],
    "fork_5.jpg": [ 0.2365196, 0.128709182, 0.5845588, 0.71405226 ],
    "fork_6.jpg": [ 0.115196079, 0.133611143, 0.676470637, 0.6993464 ],
    "fork_7.jpg": [ 0.164215669, 0.31008172, 0.767156839, 0.410130739 ],
    "fork_8.jpg": [ 0.118872553, 0.318251669, 0.817401946, 0.225490168 ],
    "fork_9.jpg": [ 0.18259804, 0.2136765, 0.6335784, 0.643790841 ],
    "fork_10.jpg": [ 0.05269608, 0.282303959, 0.8088235, 0.452614367 ],
    "fork_11.jpg": [ 0.05759804, 0.0894935, 0.9007353, 0.3251634 ],
    "fork_12.jpg": [ 0.3345588, 0.07315363, 0.375, 0.9150327 ],
    "fork_13.jpg": [ 0.269607842, 0.194068655, 0.4093137, 0.6732026 ],
    "fork_14.jpg": [ 0.143382356, 0.218578458, 0.7977941, 0.295751631 ],
    "fork_15.jpg": [ 0.19240196, 0.0633497, 0.5710784, 0.8398692 ],
    "fork_16.jpg": [ 0.140931368, 0.480016381, 0.6838235, 0.240196079 ],
    "fork_17.jpg": [ 0.305147052, 0.2512582, 0.4791667, 0.5408496 ],
    "fork_18.jpg": [ 0.234068632, 0.445702642, 0.6127451, 0.344771236 ],
    "fork_19.jpg": [ 0.219362751, 0.141781077, 0.5919118, 0.6683006 ],
    "fork_20.jpg": [ 0.180147052, 0.239820287, 0.6887255, 0.235294119 ],
    "scissors_1.jpg": [ 0.4007353, 0.194068655, 0.259803921, 0.6617647 ],
    "scissors_2.jpg": [ 0.426470578, 0.185898721, 0.172794119, 0.5539216 ],
    "scissors_3.jpg": [ 0.289215684, 0.259428144, 0.403186262, 0.421568632 ],
    "scissors_4.jpg": [ 0.343137264, 0.105833367, 0.332107842, 0.8055556 ],
    "scissors_5.jpg": [ 0.3125, 0.09766343, 0.435049027, 0.71405226 ],
    "scissors_6.jpg": [ 0.379901975, 0.24308826, 0.32107842, 0.5718954 ],
    "scissors_7.jpg": [ 0.341911763, 0.20714055, 0.3137255, 0.6356209 ],
    "scissors_8.jpg": [ 0.231617644, 0.08459154, 0.504901946, 0.8480392 ],
    "scissors_9.jpg": [ 0.170343131, 0.332957536, 0.767156839, 0.403594762 ],
    "scissors_10.jpg": [ 0.204656869, 0.120539248, 0.5245098, 0.743464053 ],
    "scissors_11.jpg": [ 0.05514706, 0.159754932, 0.799019635, 0.730392158 ],
    "scissors_12.jpg": [ 0.265931368, 0.169558853, 0.5061275, 0.606209159 ],
    "scissors_13.jpg": [ 0.241421565, 0.184264734, 0.448529422, 0.6830065 ],
    "scissors_14.jpg": [ 0.05759804, 0.05027781, 0.75, 0.882352948 ],
    "scissors_15.jpg": [ 0.191176474, 0.169558853, 0.6936275, 0.6748366 ],
    "scissors_16.jpg": [ 0.1004902, 0.279036, 0.6911765, 0.477124184 ],
    "scissors_17.jpg": [ 0.2720588, 0.131977156, 0.4987745, 0.6911765 ],
    "scissors_18.jpg": [ 0.180147052, 0.112369314, 0.6262255, 0.6666667 ],
    "scissors_19.jpg": [ 0.333333343, 0.0274019931, 0.443627447, 0.852941155 ],
    "scissors_20.jpg": [ 0.158088237, 0.04047389, 0.6691176, 0.843137264 ]
}

# This walks a directory, using the folder name as the tag name. For each folder we create a tag and then upload all
# files in that directory associating them with that tag.
tags = os.listdir(IMAGES_FOLDER)
print (f"Uploading images from {IMAGES_FOLDER}")
for tag in tags:
    # Create the tag, and grab the id
    tag_id = trainer.create_tag(project.id, tag).id
    tag_dir = os.path.join(IMAGES_FOLDER, tag)
    for image in os.listdir(tag_dir):
        with open(os.path.join(tag_dir, image), mode="rb") as img_data: 
            x,y,w,h = image_regions[image]
            print (f"Uploading: {image} with tag {tag} and region {x},{y},{w},{h}")
            regions = [ Region(tag_id=tag_id, left=x,top=y,width=w,height=h) ]
            trainer.create_images_from_files(project.id, images=[ImageFileCreateEntry(name=image, contents=img_data.read(), regions=regions)])

# Advanced training
# Advanced training is triggered by passing the advanced training type and specifying a budget.
# You can optionally specify a notification email address to receive and email when training completes.
print ("Starting Training...")
trainer.train_project(project.id, training_type=TrainingType.advanced, reserved_budget_in_hours=9, notification_email_address=NOTIFY_EMAIL)