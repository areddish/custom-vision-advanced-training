# custom-vision-advanced-training

This project is organized by project type. For each project type there are three python scripts:

- **create_and_train.py**: Creates a project, creates tags and or regions, and uploads the images to the project. Afterwards it kicks off advanced training.
- **publish.py**: Used to publish the trained iteration to a prediction resource.
- **prediction.py**: Used to send a file to the prediction resource to get results.

The flow to use this repo:

1. Clone the repo
```
git clone https://github.com/areddish/custom-vision-advanced-training
```
2. **cd** into the project type directory e.g. cd <your repo location>/image_classification
```
cd custom-vision-advanced-training/image_classification
```
3. **Update the config.py file** with your keys and information from the [Custom Vision.ai portal](https://customvision.ai)
4. Create and start advanced training for the project
```Python
python create_and_train.py
```
5. If you set a notification email in the config.py you should get an email when training is done. Otherwise you can check in the [Custom Vision.ai portal](https://customvision.ai) on the project with the name in the config.py's performance tab.
6. After the project is trained, publish it
```Python
python publish.py
```
7. Now your project is ready to predict, to use it as a predictor:
```Python
python predict.py
```