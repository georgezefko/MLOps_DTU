import click
import logging
from src.models.train_drift import train_drift

@click.command()
@click.argument('data_filepath', type=click.Path(), default='data')
@click.argument('trained_model_filepath', type=click.Path(),
                default='models/trained_model.pth')


def drift_model_command_line(data_filepath, trained_model_filepath):
    """ Trains the neural network using MNIST training data """
    _ = train_drift(data_filepath, trained_model_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    drift_model_command_line()