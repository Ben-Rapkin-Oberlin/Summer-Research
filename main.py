import numpy as np 
import pandas as pd 
from functools import partial
import os
#from filelock import FileLock

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchmetrics import Accuracy

from ray import air, tune
from ray.air import Checkpoint, session
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

#@title old net
class Net(nn.Module):
    def __init__(self, l1=12, l2=10,p=0.2):
        super(Net, self).__init__()
        self.fc1=nn.Linear(784, l1) # 16 input features, 12 output features also called neurons
        self.fc2=nn.Linear(l1, l2)
        self.out=nn.Linear(l2, 10)
        self.m = nn.LogSoftmax(dim=0)
        self.drop = nn.Dropout(p)
    
    def forward(self, x):
        x=x.squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
       
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop(x)
       
        x=self.out(x)
        x = F.relu(x)
        #x = self.drop(x)
        output = self.m(x)
        
        return output
        
def load_data(data_dir="/kaggle/working/"):
    training_data = torchvision.datasets.FashionMNIST(
    root="data_dir",
    train=True,
    download=True,
    transform=transforms.ToTensor()
    )

    test_data = torchvision.datasets.FashionMNIST(
    root="data_dir",
    train=False,
    download=True,
    transform=transforms.ToTensor()
    )
    training_data.data=torch.flatten(training_data.data, start_dim=1)
    test_data.data=torch.flatten(test_data.data, start_dim=1)
    #print(training_data.data.shape)
    return training_data, test_data
#t,te=load_data()
#print(t.data.shape)
#print(te.data.shape)

#@title accuracy
def test_accuracy(net, device="cpu"):
    _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        c=0
        for data in testloader:
                  #  c+=1
                    #if (c==2500):
                       # print('g',c)
                    #    break
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    #print(images.shape)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
    return correct / total
#print(test_accuracy(Net(16,8)))
#print(test_accuracy(Net(16,8)))
#print(test_accuracy(Net(16,8)))

def train_cifar(config,epochs=10,data_dir=None,trainset=None,testset=None,tune=False):
    net = Net(l1=config["l1"],l2=config["l2"],p=config["dropout"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    if config["opt"]=="SGD":
        #if tune:
           # config["p"]=np.random.uniform(low=.85, high=.95)
        optimizer = optim.SGD(net.parameters(), lr=config["lr"])#, momentum=.9)
        
    elif config["opt"]=="Adam":
        optimizer = optim.Adam(net.parameters(),lr=config["lr"])
        
    start_epoch = 0
    if tune:
        checkpoint = session.get_checkpoint()

        if checkpoint:
            checkpoint_state = checkpoint.to_dict()
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4
    )

    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs)
            
            #print(labels)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
           

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        if tune:
            checkpoint_data = {
                       "epoch": epoch,
                        "net_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)
            session.report(
                        {"loss": val_loss / val_steps, "accuracy": correct / total},
                        checkpoint=checkpoint,
                )
            """
            if config["opt"]=="SGD":
                session.report(
                        {"loss": val_loss / val_steps, "accuracy": correct / total},
                        checkpoint=checkpoint,
                )
            elif config["opt"]=="Adam":
                session.report(
                        {"loss": 50, "accuracy": correct / total},
                        checkpoint=checkpoint,
                )
              """
        else:
            if epoch%5==0:
                print("epoch,val loss,%correct",epoch,val_loss,correct/total)
    print("Finished Training")
    
    return net



