from azureml.core import Run
import pandas as pd
from argparse import ArgumentParser as ap
import os
from sklearn.preprocessing import MinMaxScaler

# Get the run context
run = Run.get_context()

# get the workspace
ws = run.experiment.workspace

# Import and read the input dataset
df = run.input_datasets['raw'].to_pandas_dataframe()

# Remove unneccessary columns
df_prep = df.drop(["ID"], axis =1 )

# variable to group all columns in dataframe
all_cols = df_prep.columns

# Find all nan values in dataset 
df_null = df_prep.isnull().sum()

# calculate the mode of each column
mod = df_prep.mode().iloc[0]
# group object type columns 
cols = df_prep.select_dtypes(include='object').columns
# Since nan values are in object type columns we can fill replace values with the calculated mode of that specific column
df_prep[cols] = df_prep[cols].fillna(mod)

# Replace numerical columns with nan values using mean of that column
me = df_prep.mean()
df_prep = df_prep.fillna(me)

# Create Dummy variables
df_prep = pd.get_dummies(df_prep, drop_first=True)

# Use MinMaxScaler to normalize the dataset
scale = MinMaxScaler()
colus = df.select_dtypes(include="number").columns
df_prep[colus] = scale.fit_transform(df_prep[colus])

# Get the Arguments from the pipeline job
parse = ap()
parse.add_argument('--data', type=str)
arg = parse.parse_args()

# Create the folder if it does not exist
os.makedirs(arg.data, exist_ok=True)

# Create the path to the file
path = os.path.join(arg.data, 'def_prep.csv')

# Save the file as csv
df_prep.to_csv(path,  index=False)

# Log null values
for col in all_cols:
    run.log(col, df_null[col])

# Complete the run
run.complete()