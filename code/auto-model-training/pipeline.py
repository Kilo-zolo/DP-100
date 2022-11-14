from azureml.core import Workspace
from azureml.core import Environment
from azureml.core.environment import CondaDependencies
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineData
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.core import Experiment

# Access the Workspace 
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Create custom environment 
testenv = Environment(name= "TestEnvironment")

# Create the dependencies object 
testenv_dep = CondaDependencies.create(conda_packages = ['scikit-learn',
                                                         'pandas'])

testenv.python.conda_dependencies = testenv_dep

# Register the environment
testenv.register(ws)

# Create a compute cluster 
clus_name = "test-pipe-cluster001"

# Configure the cluster 
comp_config =AmlCompute.provisioning_configuration(
                                                    vm_size='STANDARD_D11_V2',
                                                    max_nodes=2)

# Create the compute cluster                     
cluster = ComputeTarget.create(ws, clus_name, comp_config)

# Recommended since it allows the cluster to be accessible through out the entire lifetime of the pipeline
cluster.wait_for_completion()

# Create the run configuration
ronfig = RunConfiguration()

# Attach the run config to the compute cluster made abpve
ronfig.target = cluster

# Attach the run config to the environment made above so that it consists of all pur dependencies
ronfig.environment = testenv

# Define Pipeline steps

# Get the dataset from workspace
idset = ws.datasets.get('Defaults')

# Define the data variable for the data reference info with path to store data 
data = PipelineData('data', 
                    datastore = ws.get_default_datastore())

# Step 01- Data Preperation
prep_step = PythonScriptStep(name='Data Prep',
                            source_directory='/Users/Krish/dev/DP-100/code/auto-model-training/',
                            script_name='DataPrep.py',
                            inputs=[idset.as_named_input('raw')],
                            outputs=[data],
                            runconfig= ronfig,
                            arguments= ['--data', data])

# Step 02- Training
train_step = PythonScriptStep(name='Train',
                            source_directory='/Users/Krish/dev/DP-100/code/auto-model-training/',
                            script_name='Train.py',
                            inputs=[data],
                            runconfig= ronfig,
                            arguments= ['--data', data])

# Configure and Build the pipeline

# Define the steps within a variable to group and abstract the configuration of our test pipeline
steps = [prep_step,
         train_step]

test_pipeline= Pipeline(workspace = ws,
                        steps = steps)

# Create the experiment and run the pipeline

testexp = Experiment(workspace= ws,
                     name = "testpipeline")

run = testexp.submit(test_pipeline)

run.wait_for_completion(show_output=True)