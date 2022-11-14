from azureml.core import Run

run = Run.get_context()
ws = run.experiment.workspace

def prep():
    in_dset = ws.datasets.get('AdultIncome').to_pandas_dataframe()
    in_dset = in_dset.drop(['fw','edu_num'], axis=1)
    in_dset = in_dset.iloc[:1000, :]
    return in_dset
