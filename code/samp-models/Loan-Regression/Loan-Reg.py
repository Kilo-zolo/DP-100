# Import required classes from Azure ml
from azureml.core import Workspace, Run

# Acces the workspace
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Get the context of the run
run02 = Run.get_context()

# Predict the loan status using Logisitc Regression using scikit-learn
import pandas as pd

#Import the dataframe from the blob storage container in Azure
dframe = pd.read_csv("https://zohds.blob.core.windows.net/datasets-ml/Loan%20Approval%20Prediction.csv?sp=r&st=2022-11-03T06:48:43Z&se=2022-11-03T14:48:43Z&sv=2021-06-08&sr=b&sig=Lq51mQQslCFDocFhjoRjUTSJP6nU%2BtNorGLpct9O9bI%3D")
dframe.head()

# Select columns from the dataset
dframe_prep= dframe[["Married",
                    "Education",
                    "Self_Employed",
                    "ApplicantIncome",
                    "LoanAmount",
                    "Loan_Amount_Term",
                    "Credit_History",
                    "Loan_Status"]]


# Clean Missing Data
dframe_prep = dframe_prep.dropna()

# Create Dummy variables
dframe_prep= pd.get_dummies(dframe_prep, drop_first=True)

# Create X & Y variables
y = dframe_prep[["Loan_Status_Y"]]
x = dframe_prep.drop(["Loan_Status_Y"], axis=1)

# Split the data into training and test datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=1234, stratify= y)


# Build the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Fit the data to the model- train the model
model.fit(x_train,y_train.values.ravel())

# Predict the outcome using test data- Score the model
# Scored label
prediction = model.predict(x_test)

# Get the probability score- Scored Probabilities 
probability = model.predict_proba(x_test)[:, 1]

# Get the confusion matrix and the accuracy/score - Evaluate the model
from sklearn.metrics import confusion_matrix
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
run02.log("Total Observations", len(dframe))
run02.log_confusion_matrix("Confusion matrix", mdict)
run02.log("Accuracy", score)

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
sco_dset.to_csv("./outputs/LoanScore.csv",
                index=False)

# Complete the run
run02.complete()