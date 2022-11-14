from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from azureml.core import Run
import argparse
import pandas as pd
import os

# Get the parameters
parse = argparse.ArgumentParser()
parse.add_argument("--data", type=str)
arg = parse.parse_args()

# Get the run context
run = Run.get_context()

# Access the Workspace
ws = run.experiment.workspace

# Read the data passed from the previouis step
path = os.path.join(arg.data, 'def_prep.csv')
df_prep = pd.read_csv(path)

# Create X and Y dataset 
x = df_prep.drop(["Default Next Month_Yes"], axis=1)
y = df_prep[["Default Next Month_Yes"]]

# Split train and test data for x and y datasets
x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                    test_size = 0.3,
                                    random_state = 1234,
                                    stratify = y)

# import linear regression model
model = LogisticRegression()

# Fit the data to the model- train the model
model.fit(x_train,y_train.values.ravel())

# Predict the outcome using test data- Score the model
# Scored label
prediction = model.predict(x_test)

# Get the probability score- Scored Probabilities 
probability = model.predict_proba(x_test)[:, 1]

# Get the accuracy and confusion matrix for the model
matrix = confusion_matrix(y_test, prediction)
score = model.score(x_test, y_test)

# Create a confusion matrix dictionary
# To output the cm as a confusion matrix instead of an array of integers
# We need to create a schema in json format to define it

mdict =  {
       "schema_type": "confusion_matrix",
       "schema_version": "1.0.0",
       "data": {"class_labels": ["N", "Y"],
                "matrix": matrix.tolist()}
   }      

# Log the metrics
run.log("Total Observations", len(df_prep))
run.log_confusion_matrix("Confusion matrix", mdict)
run.log("Accuracy", score)

# Create and Upload the Scored dataset

# Remove the index values from the x_test and y_test dframes
# If we do not when we concatennate the dframes together their indexs will also be present and it'll have repetitve indexing
x_test = x_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Create a dframe for probabilities scored 
prob_dframe = pd.DataFrame(probability, columns=["Probabilities"])

# Create a dframe for predicitions scored 
pred_dframe = pd.DataFrame(prediction, columns=["Predictions"])

#Concatenate the above dataframes to form the score dataframe
sco_dset = pd.concat([x_test,y_test,prob_dframe,pred_dframe], axis = 1)

# Save to output folder and upload to ws
sco_dset.to_csv("./outputs/DefaultScore.csv",
                index=False)

# Complete the run
run.complete()
