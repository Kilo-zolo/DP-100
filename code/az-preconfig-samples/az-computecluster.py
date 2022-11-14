# Create a compute cluster using AzureML sdk

# Import Workspace class
from azureml.core import Workspace

# Access the workspace from the config.json file
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")

# Specify the cluster name

clus_name = "dp100cluster"

# Provisioning config for our using AmlCompute
from azureml.core.compute import AmlCompute

# Create a provisioning object to configure the compute cluster
comp_config = AmlCompute.provisioning_configuration(
                                                    vm_size= "STANDARD_D11_V2",
                                                    max_nodes= 2)

# Create a cluster
cluster = AmlCompute.create(ws, clus_name, comp_config)



