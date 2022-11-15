from azureml.core import Run
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib

run = Run.get_context()

ws = run.experiment.workspace

# Get the dataframe to use within this script passed as an argument from the job.py file
dframe = run.input_datasets['raw'].to_pandas_dataframe()
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
file = './outputs/web-scale-mods.pkl'

joblib.dump(value= [trained_cols, model], 
            filename = file)

# Complete the run
run.complete()             