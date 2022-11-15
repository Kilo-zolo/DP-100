# Import necessary libs
from azureml.core import Workspace, Dataset, Experiment, Environment
from azureml.core.environment import CondaDependencies
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import ScriptRunConfig

# Get workspace from config file
print("Getting the Workspace, so sit back, relax and let your anxiety take over 🚨...")
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Get the dataset
print("Ooooh sCaaaaRy, hope you have your dataset name right otherwise we finna get an error here 😉...")
dset= Dataset.get_by_name(ws, name='Adult')

# Create the environment 
env = Environment(name = "joblib-env")

# Create the dependencies for environment
env_dep = CondaDependencies.create(conda_packages= ['scikit-learn', 'pandas', 'joblib'])
env.python.conda_dependencies = env_dep

# Register the environment
print("Registering the environment, for legal reasons...")
env.register(ws)

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

    clust = ComputeTarget.create(ws, clust_name, conf_comp)
    clust.wait_for_completion()
else:
     clust = ws.compute_targets[clust_name]
     print(clust_name, ", already exists dickhead. You just had to make me search for it didn't you...")

# Create a script configuration for custom environment of env
scronfig = ScriptRunConfig(source_directory="/Users/Krish/dev/DP-100/code/deploy-samp/",
                            script = "web-model-script.py",
                            arguments= ['--data', dset.as_named_input('raw')],
                            environment = env,
                            compute_target= clust)


# Create and access the experiment
print("Making the experiment, hold onto your butts 🍑...")
exp = Experiment(workspace = ws, name = 'webservice-samp')

# Run the experiment 
print("Going for a run, be right back 🏃‍♂️💨...")
run = exp.submit(config= scronfig)
run.wait_for_completion(show_output=True)

# Get the run IDs from the experiment
list(exp.get_runs())