# Load the serialised object using the joblib loader
#--------------------------------------------------#

# Import the libs
from azureml.core import Workspace, Dataset
import joblib
import pandas as pd

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
d2 = dframe.copy()

# Get the numeric cols from d1
cols = d2.select_dtypes(include='number').columns

# Load/Deserialize the object
pickle = '/Users/Krish/dev/DP-100/code/deploy-samp/scaley.pkl'
load = joblib.load(pickle)

d2[cols] = load.transform(d2[cols])

# For viewing the dataset
print(d2.head())