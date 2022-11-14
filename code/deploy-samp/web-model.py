# Import necessary libs
from azureml.core import Workspace, Dataset, Experiment 
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

# Get workspace from config file
print("Getting the Workspace, so sit back, relax and let your anxiety take over 🚨...")
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Import dataset from assets store in our datastore in azure
print("Ooooh sCaaaaRy, hope you have your dataset name right otherwise we finna get an error here 😉...")
dset= Dataset.get_by_name(ws, name='Adult')
dframe= dset.to_pandas_dataframe()

# Create and access the experiment
print("Making the experiment, hold onto your butts 🍑...")
exp = Experiment(ws, 'webservice-samp')

# Run the experiment 
print("Going for a run, be right back 🏃‍♂️💨...")
run = exp.start_logging()

# Create the x and y variables
x = dframe.iloc[: , :-1]
y = dframe.iloc[:, -1:]

# Create dummy variables
x = pd.get_dummies(x)

# Extract columns inclusive of dummy vars
trained_cols = x.columns

# Transform categorical cols in y dataset to dummy
y = pd.get_dummies(y)
y = y.iloc[:,-1]

# Split data 
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.3, random_state = 1234, stratify=y)

# Build the Random Forest model
classi = RandomForestClassifier(random_state= 1234)

# Fit the data and train the model
model = classi.fit(x_train, y_train)

# Predict the outcome using test data
pred = classi.predict(x_test)

# Calculate the probaility scores
prob = classi.predict_proba(x_test)[: , 1]

# Get Confusion matrix and the accuracy - Evaluate
matrix = confusion_matrix(y_test, pred)
score = classi.score(x_test, y_test)

# Always log the primary metric
run.log("accuracy", score)

# Save all models and transformations
file = '/Users/Krish/dev/DP-100/code/deploy-samp/outputs/web-scale-mods.pkl'

joblib.dump(value= [trained_cols, model], 
            filename = file)

# Complete the run
run.complete()            
