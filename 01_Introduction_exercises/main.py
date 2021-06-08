import sys
import argparse

import torch
from torch import nn,optim
import numpy as np
import matplotlib.pyplot as plt


from data import mnist
from model import MyAwesomeModel

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        
        
        parser.add_argument("command",help="Subcommand to run")
        
        
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001,type = int)
        parser.add_argument('--epochs',default=20, type = int)
        parser.add_argument('--input_size',default=784, type = int)
        parser.add_argument('--hidden_size',default=256, type = int)
        parser.add_argument('--output',default=10, type = int)
        parser.add_argument('--batch_size',default=64, type = int)
        
        
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # Implement training loop here
        model = MyAwesomeModel(args.input_size,args.hidden_size,args.output)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), args.lr)
        train_loader,val_loader, train_set, val_set, _,_ = mnist(args.batch_size)

        
        
        
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for e in range(args.epochs):
            train_loss = 0
            train_correct = 0
            
            for images,labels in train_loader:

                model.train()
                optimizer.zero_grad()

                output = model(images)
                ps     = torch.exp(output)
                loss = criterion(output,labels)

                loss.backward()

                optimizer.step()
                train_loss += loss.item()
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                train_correct += equals.type(torch.FloatTensor).sum().item()  

                
            else:
                # Compute validattion loss and accuracy
                val_loss    = 0
                val_correct = 0
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    model.eval()     # Sets the model to evaluation mode
                    for images, labels in val_loader:
                        # Forward pass and compute loss
                        output    = model(images)
                        ps        = torch.exp(output)
                        val_loss += criterion(output, labels)
                        
                        # Keep track of how many are correctly classified
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        val_correct += equals.type(torch.FloatTensor).sum().item()
                
                # Store and print losses and accuracies
                train_losses.append(train_loss/len(train_loader))
                train_accuracies.append(train_correct/len(train_set))
                val_losses.append(val_loss/len(val_loader))
                val_accuracies.append(val_correct/len(val_set))

                print("Epoch: {}/{}.. ".format(e+1, args.epochs),
                      "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                      "Training Accuracy: {:.3f}".format(train_accuracies[-1]),
                      "Validation Loss: {:.3f}.. ".format(val_losses[-1]),
                      "Validation Accuracy: {:.3f}".format(val_accuracies[-1]))

        # Save the trained network
        torch.save(model.state_dict(), 'trained_model.pth')
                
        # Plot the training loss curve
        print('start ploting')
        f = plt.figure(figsize=(12, 8))
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses,   label='Validation loss')
        plt.xlabel('Epoch number')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        f.savefig('/Users/georgioszefkilis/dtu_mlops-new/01_introduction/final_exercise/Training_Loss.pdf', bbox_inches='tight')
        
        # Plot the training accuracy curve
        f = plt.figure(figsize=(12, 8))
        plt.plot(train_accuracies, label='Training accuracy')
        plt.plot(val_accuracies,   label='Validation accuracy')
        plt.xlabel('Epoch number')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        f.savefig('/Users/georgioszefkilis/dtu_mlops-new/01_introduction/final_exercise/Training_Accuracy.pdf', bbox_inches='tight')
        
                
        
        
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        parser.add_argument('--input_size',default=784, type = int)
        parser.add_argument('--hidden_size',default=256, type = int)
        parser.add_argument('--output',default=10, type = int)
        parser.add_argument('--batch_size',default=64, type = int)

        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # Implement evaluation logic here
        model = MyAwesomeModel(args.input_size,args.hidden_size,args.output)
        if args.load_model_from:
            #model = torch.load(args.load_model_from)
            state_dict = torch.load(args.load_model_from)
            model.load_state_dict(state_dict)

        _,_, _,_,test_loader,test_set = mnist(args.batch_size)
        
        
        test_correct =0 
        with torch.no_grad():
            model.eval()
            
            
            for images,labels in test_loader:

                output = model(images)
                ps = torch.exp(output)

                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_correct += equals.type(torch.FloatTensor).sum().item()
            test_accuracy = test_correct/len(test_set)
        print("Test Accuracy: {:.3f}".format(test_accuracy))
                
        

if __name__ == '__main__':
    
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    
    
    
# %%
