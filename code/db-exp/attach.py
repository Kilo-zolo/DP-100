# Attach the databricks cluster to the AzureML Workspace as an attached compute

# Import the necessary libraries
from azureml.core import Workspace, Environment, Experiment
from azureml.core.compute import DatabricksCompute, ComputeTarget, AmlCompute
from azureml.core.environment import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import DatabricksStep, PythonScriptStep
from azureml.core.databricks import PyPiLibrary

# Access the Workspace
print("Accessing Workspace...")
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Create the custom environment
print("Creating the environment...")
env = Environment(name="az-db-env")

# Create the dependency object
print("Adding dependencies...")
env_dep = CondaDependencies.create(conda_packages=['scikit-learn', 'joblib', 'pandas'])
env.python.conda_dependencies = env_dep

# Register the environment
print("Registering the environment...")
env.register(ws)

# Create the compute cluster for the pipeline
pipe_comp = "pipeCluster"
# Provision the config using AmlCompute
print("Accessing the compute cluster for the pipeline...")

if pipe_comp not in ws.compute_targets:
    print("Creating the pipeline cluster named: ", pipe_comp)
    pionfig = AmlCompute.provisioning_configuration(
                                                    vm_size= "STANDARD_D11_V2",
                                                    min_nodes=1,
                                                    max_nodes=2)

    pipe_clust = AmlCompute.create(ws, pipe_comp, pionfig)
    pipe_clust.wait_for_completion()
else:
    pipe_clust = ws.compute_targets[pipe_comp]
    print(pipe_comp, ", compute cluster found, using it...")

# Create the Run Configuration for these steps
print("Creating the run configuration...")
ronfig = RunConfiguration()

ronfig.target = pipe_clust
ronfig.environment = env

# Creat config for the databrick cluster
print("Initializing Databricks compute cluster parameters...")
dbrick_rg = "az-db-rg"
dbrick_ws = "Dbricks"
dbrick_token = "dapieca5db7a7fb080f1b099b27b351a9fe9"
dbrick_comp = "dbricksCluster"

if dbrick_comp not in ws.compute_targets:
    print("Creating the Databricks compute cluster...")
    attonfig = DatabricksCompute.attach_configuration(
                                                            resource_group= dbrick_rg,
                                                            workspace_name= dbrick_ws,
                                                            access_token= dbrick_token)
    print("Attaching the compute cluster...")
    dbrick_clust = ComputeTarget.attach(ws, dbrick_comp, attonfig)

    dbrick_clust.wait_for_completion(True)

else:
    print("Attaching the compute cluster...")
    dbrick_clust = ws.compute_targets[dbrick_comp]    

# Create and pass the data reference of Input and Output
print("Getting the datastore...")
dstore = ws.datastores.get('adultincome')

inata = DataReference(datastore= dstore,
                      data_reference_name= 'in')

outata1 = PipelineData('outata', datastore=dstore)


# Create the Databricks steps

# Creating the libraries needed to be added to the databricks step
skl = PyPiLibrary(package= 'scikit-learn')
jblb = PyPiLibrary(package= 'joblib')

# Location of the notebook we will be utilising in the step
note_path = r"/Users/dr.girish.gupta@outlook.com/demo"

dbrick_step01 = DatabricksStep(name = "prep",
                               inputs = [inata],
                               outputs = [outata1],
                               num_workers = 1,
                               notebook_path = note_path,
                               run_name = "db_demo", 
                               compute_target = dbrick_clust,
                               pypi_libraries = [skl, jblb],
                               allow_reuse = False)

# Create the pipeline step to run the python script
                               
evalu_step =  PythonScriptStep(name= 'Evaluation',
                                source_directory= '/Users/Krish/dev/DP-100/code/db-exp',
                                script_name= 'evaluate.py',
                                inputs=[outata1],
                                runconfig=ronfig,
                                arguments=['--outata', outata1])

# Build and submit the pipeline
steps = [dbrick_step01, evalu_step]
pipeline = Pipeline(workspace = ws, steps = steps)
pipe_run = Experiment(ws, 'db-note-exp').submit(pipeline)                                

# Wait for all steps completion
pipe_run.wait_for_completion(show_output=True)