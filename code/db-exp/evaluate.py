# Model EValuation step for the Pipelin run

# Import the required libs
from azureml.core import Run
import argparse
import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import sklearn.ensemble._forest

# Get the context of the experiment run
run = Run.get_context()

# Access the workspace
ws = run.experiment.workspace
print("Workspace accessed...")

# Get Parameters
parse = argparse.ArgumentParser()
parse.add_argument("--outata", type=str)
args = parse.parse_args()
print("arguments accessed and printing...")
print(args.outata)

# Read the data from the previous step
path = os.path.join(args.outata, 'x_test.csv')
x_test = pd.read_csv(path)

path = os.path.join(args.outata, 'y_test.csv')
y_test = pd.read_csv(path)

path = os.path.join(args.outata, 'predictions.csv')
pred = pd.read_csv(path)

obj = os.path.join(args.outata, 'ClassiModel.pkl')
classi = joblib.load(obj)
score = classi.score(x_test, y_test)

# Evaluate the model
matrix = confusion_matrix(y_test, pred)

dict = {"schema_type": "confusion_matrix",
        "schema_version": "v1",
        "data": {"class_labels": ["N", "Y"],
                 "matrix": matrix.tolist()}
        }

run.log_confusion_matrix("ConfusionMatrix", dict)
run.log("Score", score)

# Complete the run
run.complete()



