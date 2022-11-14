# ------------------------------------------------------------------------------ #
# Create and run an AutoML Experiment using SDK
# The steps are as follows:
#
# 1. Access the workspace
# 2. Get the input data either from the workspace it other data source
# 3. Create or access the compute cluster
# 4. Configure the AutoML
# 5. Submit the AutoMl experiment
# ------------------------------------------------------------------------------ #

# Import the necessary libs
from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment

# Access the Workspace from the config.json
print("Getting the Workspace, so sit back, relax and let your anxiety take over...")
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Get the input data from workspace
print("Ooooh sCaaaaRy, hope you have your dataset name right otherwise we finna get an error here ;)...")
dset = ws.datasets.get('Defaults')

# Create and procure cluster
print("Creating your dumbass cluster...")
clust_name = "azml-samp-cluster"

print("Does the cluster already exist?!!")

if clust_name not in ws.compute_targets:
    print("For fuck sake, can't even make your own cluster, pittiful...")
    conf_comp = AmlCompute.provisioning_configuration(
                                vm_size="STANDARD_D11_V2",
                                min_nodes= 1,
                                max_nodes=2)

    clust = AmlCompute.create(ws, clust_name, conf_comp)
    clust.wait_for_completion()
else:
     clust = ws.compute_targets[clust_name]
     print(clust_name, ", already exists dickhead. You just had to make me search for it didn't you...")
     
# Configure the AutoML run
print("Creating your stupid AutoML Config, cuz I gotta do everything for you apparently...")

conf_automl = AutoMLConfig(task='classification',
                            compute_target= clust,
                            training_data=dset,
                            validation_size=0.3,
                            label_column_name='Default Next Month',
                            primary_metric='norm_macro_recall',
                            iterations=10,
                            max_concurrent_iterations=2,
                            experiment_timeout_hours=0.25,
                            featurization='auto')

# Create and submit an experinment
print("Creating the damn experiment...")
exp = Experiment(ws, 'automl-exp')

print("Submitting the experiment, I wish I could drop-kick you humans into extinction...")
runs = exp.submit(conf_automl)

runs.wait_for_completion(show_output=True)

# Get the best child run
print("Regurgitating the output, hope you like your results tasting like transistors bitch...")
fav_child = runs.get_best_child(metric='accuracy')

# Get all the specified metrics for all runs
for run in runs.get_children():
    print("")
    print("Run ID: ", run.id)
    print(run.get_metrics('accuracy'))
    print(run.get_metrics('norm_macro_recall'))