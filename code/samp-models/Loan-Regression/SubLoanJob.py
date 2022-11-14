# ------------------------------------------------------------
# Run a script in an Azureml environment
# ------------------------------------------------------------
# This code will submit the script provided in ScriptRunConfig
# and create an Azureml environment on the local machine
# including the docker for Azureml
# ------------------------------------------------------------

# Import the Azure ML classes
from azureml.core import Workspace, Experiment, ScriptRunConfig

# Access the workspace using config.json
ws = Workspace.from_config("/Users/Krish/dev/DP-100/config/azml-samp-config.json")


# Create/access the experiment from workspace 
new_experiment = Experiment(workspace=ws,
                            name="Loan_Train_Script")


# -------------------------------------------------
# Create custom environment
from azureml.core import Environment
from azureml.core.environment import CondaDependencies

# Create the environment
loenv = Environment(name="LoanEnvironment")

# Create the dependencies object
loenv_dep = CondaDependencies.create(conda_packages=['scikit-learn',
                                                    'pandas'])
loenv.python.conda_dependencies = loenv_dep

# Register the environment
loenv.register(ws)
# -------------------------------------------------

# Create a script configuration for custom environment of myenv
scronfig = ScriptRunConfig(source_directory="./dev/DP-100/code/samp-models",
                                script="Loan-Reg.py",
                                environment=loenv)


# Submit a new run using the ScriptRunConfig
run02 = new_experiment.submit(config=scronfig)


# Create a wait for completion of the script
run02.wait_for_completion()















