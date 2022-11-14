# Decision Tree Classifier 
# Predict the income of an Adult based on the census data

# Import libs
from azureml.core import Workspace, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from interpret.ext.blackbox import TabularExplainer

# Get workspace from config file
print("Getting the Workspace, so sit back, relax and let your anxiety take over...")
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Import dataset from assets store in our datastore in azure
print("Ooooh sCaaaaRy, hope you have your dataset name right otherwise we finna get an error here ;)...")
dset= Dataset.get_by_name(ws, name='Adult')
dframe= dset.to_pandas_dataframe()

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

# Get the feature importance data
print("Hope your computer isnt on fire cuz im about to spit some straight bars...")
imp_feat = globe.get_feature_importance_dict()
print("Importance of features based on their Shapley values:")
for i in imp_feat:
    print(i)

# Get the local feature importance
imp_ind = x_test[0:5]

local = explainer.explain_local(imp_ind)

local_feats = local.get_ranked_local_names()
local_imp = local.get_ranked_local_values()

# Create lopp to print the local names and their ranked local values

for i in range(0, len(local_feats)):
    labes = local_feats[i]
    print("\n Feature Support Values for : ", classes[i])

    for j in range(0, len(labes)):

        if pred[j] == i:
            print("\n\tObservation number : ", j + 1)
            feat_names = labes[j]

            print("\t\t", "Feature Name".ljust(30), " Value")
            print("\t\t", "-"*30, "-"*10)

            for k in range(0, len(feat_names)):
                print("\t\t", feat_names[k].ljust(30), round(local_imp[i][j][k], 6))
