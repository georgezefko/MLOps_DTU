import logging
import pickle
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import torch
from src.models.model import MyAwesomeModel
from torch import nn, optim
from torchvision import datasets, transforms
import torchdrift
import sklearn.manifold
import math
from matplotlib import pyplot
import click

@click.command()
@click.argument('data_filepath', type=click.Path(), default='data')
@click.argument('trained_model_filepath', type=click.Path(),default='models/trained_model.pth')
@click.argument('training_figures_filepath', type=click.Path(),
                default='reports/figures/')







def drift(data_filepath, trained_model_filepath,training_figures_filepath):
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Drift on MINST')

    # Import model
    model = MyAwesomeModel()
    project_dir = Path(__file__).resolve().parents[2]
    state_dict = torch.load(project_dir.joinpath(trained_model_filepath))
    model.load_state_dict(state_dict) 
    

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)), ])

    # Divide the training dataset into two parts:
    #  a training set and a validation set
    project_dir = Path(__file__).resolve().parents[2]
    train_set = datasets.MNIST(project_dir.joinpath(data_filepath),
                               download=False, train=True,
                               transform=transform)
    batch_size = 64
    train_n = int(0.7*len(train_set))
    val_n = len(train_set) - train_n
    train_set, val_set = torch.utils.data.random_split(train_set,
                                                       [train_n, val_n])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=True)

    #show inputs
    inputs, _ = next(iter(val_loader))
    inputs_ood = corruption_function(inputs)


    feature_extractor = copy.deepcopy(model)
    feature_extractor.classifier = torch.nn.Identity()

    #drift detector
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    torchdrift.utils.fit(train_loader, feature_extractor, drift_detector, num_batches= 10)

    drift_detection_model = torch.nn.Sequential(
    feature_extractor,   drift_detector)
    features = feature_extractor(inputs)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    print ('Score',score)
    print ("Pvalue", p_val)


    N_base = drift_detector.base_outputs.size(0)
    mapper = sklearn.manifold.Isomap(n_components=2)
    base_embedded = mapper.fit_transform(drift_detector.base_outputs)
    features_embedded = mapper.transform(features.detach().numpy())
    f = plt.figure(figsize=(12, 8))
    pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    pyplot.title(f'score {score:.2f} p-value {p_val:.2f}')
    f.savefig(project_dir.joinpath(training_figures_filepath).joinpath('drift.pdf'),
              bbox_inches='tight')
    


    features = feature_extractor(inputs_ood)
    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)

    features_embedded = mapper.transform(features.detach().numpy())
    f = plt.figure(figsize=(12, 8))
    pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    pyplot.title(f'score {score:.2f} p-value {p_val:.2f}')
    f.savefig(project_dir.joinpath(training_figures_filepath).joinpath('drift2.pdf'),
              bbox_inches='tight')

def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    drift()