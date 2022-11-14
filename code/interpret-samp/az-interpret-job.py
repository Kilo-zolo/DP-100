# Decision Tree Classifier 
# Predict the income of an Adult based on the census data

# Import libs
from azureml.core import Workspace, Dataset, Experiment
from azureml.core import Environment 
from azureml.core.environment import CondaDependencies
from azureml.core.compute import AmlCompute
from azureml.core import ScriptRunConfig

# Get workspace from config file
print("Getting the Workspace, so sit back, relax and let your anxiety take over...")
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Import dataset from assets store in our datastore in azure
print("Ooooh sCaaaaRy, hope you have your dataset name right otherwise we finna get an error here ;)...")
dset= Dataset.get_by_name(ws, name='Adult')

# Create the environment
env = Environment(name="az-samp-env")

# Create the dependencies object
env_dep = CondaDependencies.create(conda_packages=['scikit-learn', 'pip'],
                                    pip_packages=['azureml-defaults', 'azureml-interpret'])
env.python.conda_dependencies = env_dep

# Register the environment
print("Registering your bitch ass environment...")
env.register(ws)

# Create and procure the compute cluster
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

# Create a ScrtipRunConfig
print("ScriptRunConfig initializing beep bop boop boop, lmao jk aint a fkn robot...")
scronfig = ScriptRunConfig(source_directory="/Users/Krish/dev/DP-100/code/interpret-samp/",
                          script="az-interpret-script.py",
                          arguments = ['--data', dset.as_named_input('raw')],
                          environment=env,
                          compute_target= clust)

# Create the experiment and run
print("Creating the damn experiment...")                     
exp = Experiment(workspace=ws, name='az-interpret')

print('Calm yo titatas, im submitting the experiment...')
run = exp.submit(config=scronfig)

run.wait_for_completion(show_output=True)


