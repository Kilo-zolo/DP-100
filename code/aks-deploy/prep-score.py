# Entry script for the Adult Income Prediction Service
import joblib
from azureml.core.model import Model
import json
import pandas as pd

# Create the init() function to unpack and load the necessary files for the model and the service deployment
def init():
    global ref, pred
    path = Model.get_model_path('AdultIncomePredictor')
    ref, pred = joblib.load(path)

# Create a run() function to perfomr necessary transformations and predictions
def run(raw):
    dict = json.loads(raw)['data']
    dset = pd.DataFrame.from_dict(dict)
    encoded = pd.get_dummies(dset)
    deploy = encoded.columns

    missing = ref.differenc(deploy)

    for cols in missing:
        encoded[cols] = 0

    encoded = encoded[ref]

    predictions = pred.predict(encoded)

    classes = ['> 50k', '< 50k']

    pred_classes = []

    for prediction in predictions:
        pred_classes.append(classes[prediction])

    return json.dumps(pred_classes)     