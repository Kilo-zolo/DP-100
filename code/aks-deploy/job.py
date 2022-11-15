# Import the libs necessary
from azureml.core import Workspace, Environment
from azureml.core.environment import CondaDependencies
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AksWebservice

# Access the workspace
print("Accessing the workspace...")
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Create the environment
env = Environment(name="env")

# Create the dependencies object
print("Creating the dependencies...")
env_dep = CondaDependencies.create(conda_packages=['scikit-learn', 'pip'],
                                    pip_packages=['azureml-defaults'])
env.python.conda_dependencies = env_dep

# Register the environment
print("Registering the environment...")
env.register(ws)

# Create an Azure Kubernetes Services provisioning configuration
name = 'aks-cluster'

akonfig = AksCompute.provisioning_configuration(location='eastaustralia',
                                                vm_size='STANDARD_D11_V2',
                                                agent_count=1,
                                                cluster_purpose='Staging')

cluster = ComputeTarget.create(ws, name, akonfig)                                                
cluster.wait_for_completion(show_output=True)

# Create the inference config
# Software and packages related configuration
infonfig = InferenceConfig(environment=env,
                           entry_script='prep-score.py',
                           source_directory='/Users/Krish/dev/DP-100/code/aks-deploy')

# Create the AKS deployment configuration
# Hardware configuration
deponfig = AksWebservice.deploy_configuration(cpu_cores=1,
                                              memory_gb=1)

# Deploy the webservice
model = ws.models['AdultIncomePredictor']                                              
serv = Model.deploy(workspace= ws,
                    name= 'adultincome-serv',
                    models= [model],
                    inference_config= infonfig,
                    deployment_config= deponfig,
                    deployment_target= cluster)

serv.wait_for_completion(show_output=True)                    