import torch
from torchvision import datasets, transforms

def mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data

    train_set = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    train_n = int(0.7*len(train_set))
    val_n   = len(train_set) - train_n
    #split train, val
    train_set, val_set = torch.utils.data.random_split(train_set, [train_n, val_n])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True) 

    # Download and load the test data
    test_set = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader,val_loader, train_set,val_set,test_loader,test_set
