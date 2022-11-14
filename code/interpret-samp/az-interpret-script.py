from azureml.core import Run
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from interpret.ext.blackbox import TabularExplainer
from azureml.interpret import ExplanationClient

# Get the run context
run = Run.get_context()

# Get workspace from the run
ws = run.experiment.workspace

# Get parameters
parse = argparse.ArgumentParser()
parse.add_argument("--data", type=str)

# Load the dataset from files
dframe = run.input_datasets['raw'].to_pandas_dataframe()

# Drop uneccessary columns
print("Dropping 'em columns like flies...")
dframe= dframe.drop(['fw','edu_num','occ','rel','cg','cl','nc'], axis=1)

# Get dummy variables
prep = pd.get_dummies(dframe, drop_first=True)

# Create x dataset and y dataset
x = prep.iloc[ : , :-1]
y = prep.iloc[ : , -1]

# Create training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1234, stratify= y) 

# Import and train Random Forest Classifier
print("Waxing on and Waxing off...")
classi = RandomForestClassifier(random_state=1234)
trained = classi.fit(x_train, y_train)

# Test our model
print("Time for the ultimate test!!")
pred = classi.predict(x_test)

# Evaluate the model 
matrix = confusion_matrix(y_test, pred)
score = classi.score(x_test,y_test)

# Always log the primary metric
run.log("accuracy", score)

# Create Explanations for the model
# Poor Bitches = > 50k income earners
# Gucci Gucci = < 50k income earners
classes = ['Poor bitches', 'Gucci Gucci']
feats = list(x.columns)

explainer = TabularExplainer(trained,
                             x_train,
                             features=feats,
                             classes=classes)

# Get Global explanations
print("Trying to explain it to you in small brain language...")
globe = explainer.explain_global(x_train)                           

# Upload the explanations to workspace
# Create client object for explanation
client = ExplanationClient.from_run(run)
client.upload_model_explanation(globe, 
                                comment = 'Adult Income Global TabularExplainer')

run.complete()                                