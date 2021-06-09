import logging
import pickle
from pathlib import Path
import sys
import argparse

import click
import matplotlib.pyplot as plt
import torch
import torchvision
from model import MyAwesomeModel
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


@click.command()
@click.argument('data_filepath', type=click.Path(), default='data')
@click.argument('trained_model_filepath', type=click.Path(),
                default='models/trained_model.pth')
@click.argument('training_statistics_filepath', type=click.Path(),
                default='data/processed/')
@click.argument('training_figures_filepath', type=click.Path(),
                default='reports/figures/')

def main(data_filepath, trained_model_filepath, training_statistics_filepath,
         training_figures_filepath):
    """ Trains the neural network using MNIST training data """
    logger = logging.getLogger(__name__)
    logger.info('Training a neural network using MNIST training data')
    writer = SummaryWriter()

    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=0.001,type = int)
    parser.add_argument('--epochs',default=20, type = int)
    parser.add_argument('--input_size',default=784, type = int)
    parser.add_argument('--hidden_size',default=256, type = int)
    parser.add_argument('--output',default=10, type = int)
    parser.add_argument('--batch_size',default=64, type = int)
        
        
        # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
        
    # Implement training loop here
     

    # Create the network and define the loss function and optimizer
    model = MyAwesomeModel(args.input_size,args.hidden_size,args.output)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    train_set_targets = train_set.targets.numpy()

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=True)
    
    # Plot example images
    images, labels = iter(train_loader).next()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i][0], cmap='gray')
    grid = torchvision.utils.make_grid(images)
    writer.add_image('MNIST examples', grid, 0)

    # Plot the data distribution of the MNIST training set
    writer.add_histogram('MNIST data distribution', train_set_targets)

    # Add graph data to summary
    writer.add_graph(model, images)

    # Implement the training loop
    epochs = 10
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for e in range(epochs):
        train_loss = 0
        train_correct = 0

        for images, labels in train_loader:
            # Set model to training mode and zero
            #  gradients since they accumulated
            model.train()
            optimizer.zero_grad()

            # Make a forward pass through the network to get the logits
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # Use the logits to calculate the loss
            loss = criterion(log_ps, labels)
            train_loss += loss.item()

            # Perform a backward pass through the network
            #  to calculate the gradients
            loss.backward()

            # Take a step with the optimizer to update the weights
            optimizer.step()

            # Keep track of how many are correctly classified
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_correct += equals.type(torch.FloatTensor).sum().item()
        else:
            # Compute validattion loss and accuracy
            val_loss = 0
            val_correct = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()     # Sets the model to evaluation mode
                for images, labels in val_loader:
                    # Forward pass and compute loss
                    log_ps = model(images)
                    ps = torch.exp(log_ps)
                    val_loss += criterion(log_ps, labels)

                    # Keep track of how many are correctly classified
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_correct += equals.type(torch.FloatTensor).sum().item()

            # Store and print losses and accuracies
            train_losses.append(train_loss/len(train_loader))
            train_accuracies.append(train_correct/len(train_set))
            val_losses.append(val_loss/len(val_loader))
            val_accuracies.append(val_correct/len(val_set))

            logger.info(str("Epoch: {}/{}.. ".format(e+1, epochs)) +
                        str("Training Loss: {:.3f}.. ".format(train_losses[-1])) +
                        str("Training Accuracy: {:.3f}.. ".format(train_accuracies[-1])) +
                        str("Validation Loss: {:.3f}.. ".format(val_losses[-1]))         +
                        str("Validation Accuracy: {:.3f}.. ".format(val_accuracies[-1])))
            
            # Write the training 
            writer.add_scalar('Loss/train', train_loss/len(train_loader), e+1)
            writer.add_scalar('Loss/validation', val_loss/len(val_loader), e+1)
            writer.add_scalar('Accuracy/train', train_correct/len(train_set), e+1)
            writer.add_scalar('Accuracy/validation', val_correct/len(val_set), e+1)

    # Save the trained network
    torch.save(model.state_dict(), project_dir.joinpath(trained_model_filepath))

    # Save the training and validation losses and accuracies as a dictionary
    train_val_dict = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }

    with open(project_dir.joinpath(training_statistics_filepath).joinpath('train_val_dict.pickle'), 'wb') as f:
        # Pickle the 'train_val_dict' dictionary using
        #  the highest protocol available
        pickle.dump(train_val_dict, f, pickle.HIGHEST_PROTOCOL)

    # Plot the training loss curve
    f = plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses,   label='Validation loss')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    f.savefig(project_dir.joinpath(training_figures_filepath).joinpath('Training_Loss.pdf'),
              bbox_inches='tight')

    # Plot the training accuracy curve
    f = plt.figure(figsize=(12, 8))
    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(val_accuracies,   label='Validation accuracy')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    f.savefig(project_dir.joinpath(training_figures_filepath).joinpath('Training_Accuracy.pdf'),
              bbox_inches='tight')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
    
    
    
    
    
    
    
    
    
    
# %%
