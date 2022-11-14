# Create serialized object for the joblib dump
#----------------------------------------------#
import joblib
from sklearn.preprocessing import MinMaxScaler
from azureml.core import Workspace, Dataset

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

# A standard practice to copy the dframe and use the copy for pre-processing so the original dataset is not entirely affected
d1 = dframe.copy()

# MinMaxScalar
scale = MinMaxScaler()

# Get the numeric cols from d1

cols = d1.select_dtypes(include='number').columns

# Fit the data to the scaler object
scaled_fit = scale.fit(d1[cols])

# Transform the data using the object we made
d1[cols] = scaled_fit.transform(d1[cols])

# Create the serialized object of the fitted scaler object
# We do this to convert the format into binary and then allow azure to revert it back to its original form 
# This allows us to provide the right formatting for a pipeline when we want todeploy it as a web-service
# Start by specifiying the pkl files path (pkl file = serialised file)
pickle = '/Users/Krish/dev/DP-100/code/deploy-samp/scaley.pkl'

# dump the object
joblib.dump(value=scaled_fit, filename=pickle)

# View the dataframe
print(d1.head())