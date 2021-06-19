import os

from azureml.core import (Environment, Experiment, Model, ScriptRunConfig,
                          Workspace)
from azureml.core.conda_dependencies import CondaDependencies


def main():
    # Create a Python environment for the experiment
    env = Environment("experiment-MNIST")

    # Load the workspace from the saved config file
    ws = Workspace.from_config()
    print('Ready to use Azure ML to work with {}'.format(ws.name))

    # Ensure the required packages are installed
    packages = CondaDependencies.create(conda_packages=['pip'],
                                        pip_packages=['azureml-defaults',
                                                      'torch', 'torchvision',
                                                      'pandas', 'numpy',
                                                      'matplotlib'])  #, '-e .'])
    whl_path = '/Users/georgioszefkilis/MLOps_DTU-1/dist/src-0.1.0-py3-none-any.whl"
    whl_url = Environment.add_private_pip_wheel(workspace=ws, exist_ok=True,
                                                file_path=whl_path)
    packages.add_pip_package(whl_url)
    env.python.conda_dependencies = packages

    # From a pip requirements file
    #   env = Environment.from_pip_requirements(name = "experiment-MNIST",
    #             file_path = "C:/Github/MLOpsExercises/requirements.txt")          

    # Create a script config for making the data set
    script_config = ScriptRunConfig(source_directory='./src/data',
                                    script='make_dataset_command_line.py',
                                    environment=env)

    # Create and submit the experiment
    experiment = Experiment(workspace=ws, name='MNIST-make-dataset')
    run = experiment.submit(config=script_config)
    run.wait_for_completion()
    print('Finished running the make dataset script')

    # Create a script config for training
    experiment_folder = './src/models'
    e = 3
    lr = 0.001
    script_args = [
        '--epochs',
        e,
        '--learning_rate',
        lr
    ]
    script_config = ScriptRunConfig(source_directory=experiment_folder,
                                    script='train_model_command_line_azure.py',
                                    environment=env, arguments=script_args)

    # Create and submit the experiment
    experiment = Experiment(workspace=ws, name='MNIST-training-14-06-2021')
    run = experiment.submit(config=script_config)

    # Block until the experiment run has completed
    run.wait_for_completion()
    print('Finished running the training script')

    # Get logged metrics and files
    print('Getting run metrics')
    metrics = run.get_metrics()
    for key in metrics.keys():
        print(key, metrics.get(key))
    print('\n')

    print('Getting run files')
    for file in run.get_file_names():
        print(file)

    # Register the model
    model_props = {'epochs': e, 'learning_rate': lr,
                   'Final train loss': metrics['Train loss'][-1],
                   'Final train accuracy': metrics['Train accuracy'][-1],
                   'Final validation loss': metrics['Validation loss'][-1],
                   'Final validation accuracy': metrics['Validation accuracy'][-1]}
    run.register_model(model_path='./outputs/models/trained_model.pth', 
                       model_name='MNIST',
                       tags={'Training data': 'MNIST'},
                       properties=model_props)

    # List registered models
    for model in Model.list(ws):
        print(model.name, 'version:', model.version)
        for tag_name in model.tags:
            tag = model.tags[tag_name]
            print('\t', tag_name, ':', tag)
        for prop_name in model.properties:
            prop = model.properties[prop_name]
            print('\t', prop_name, ':', prop)
        print('\n')

    # Download files in the "outputs" folder and store locally
    download_folder = 'azure-downloaded-files'
    run.download_files(prefix='outputs', output_directory=download_folder)

    # Verify the files have been downloaded
    for root, directories, filenames in os.walk(download_folder):
        for filename in filenames:
            print(os.path.join(root, filename))

if __name__ == '__main__':
    main()