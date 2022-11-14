# Handle dummy variables in production
# ------------------------------------#

import pandas as pd
import joblib

# Create a dataframe
x_train = pd.DataFrame({'name'      : ['Krish', 'Sarah', 'Moon', 'Usuf'],
                        'income'    : [90, 40, 50, 70],
                        'occupation': ['Corpo', 'Corpo', 'Non-Corpo', 'Non-Corpo']})

# Create dummy variables
ex_train = pd.get_dummies(x_train)

# Extract col names as the index object
encoded = ex_train.columns

# Serialize the column index
file = '/Users/Krish/dev/DP-100/code/deploy-samp/dummy-vars/cols.pkl'
joblib.dump(value= encoded, filename= file)

# Production usage of index object
cols_refer = joblib.load(file)

# Example input data for webservice
x_deploy = pd.DataFrame({'name' : ['Chaitra'],
                        'income': [70],
                        'occupation' : ["Non-Corpo"]})

# Create dummy variables
ex_deploy = pd.get_dummies(x_deploy)

# Extract column names of prod data
cols_deploy = ex_deploy.columns

# Find the missing cols 
cols_miss = cols_refer.difference(cols_deploy)

# Add the missing cols
for cols in cols_miss:
    ex_deploy[cols] = 0

# Ensure the cols are in the same order as the pre-deployed dataframe
ex_deploy = ex_deploy[cols_refer]
print(ex_deploy)

# Adding an unknown category
cols_xtra = cols_deploy.difference(cols_refer)

if len(cols_xtra) != 0:
    print(" You have an unknown category present : ", cols_xtra)