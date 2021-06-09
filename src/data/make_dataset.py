# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import torch
from torchvision import datasets, transforms




def data(data_filepath):
    """ Downloads and stores the MNIST training and test data into
        raw data (../DATA_FILEPATH/MNIST/raw) and into cleaned data ready
        to be analyzed (../DATA_FILEPATH/MNIST/processed)
    """

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)), ])

    logger = logging.getLogger(__name__)
    logger.info('Download and store the training and test data')
    project_dir = Path(__file__).resolve().parents[2]
    train = datasets.MNIST(project_dir.joinpath(data_filepath),
                       download=True, train=True,
                       transform=transform)
    test = datasets.MNIST(project_dir.joinpath(data_filepath),
                       download=True, train=False,
                       transform=transform)

    return train, test
