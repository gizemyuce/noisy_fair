import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

import wandb
import os
import numpy as np

import math

import torchvision.models as models


class PolynomialLoss(nn.Module):
    """
    Poly-tailed margin based losses that decay as v^{-\alpha} for \alpha > 0.
    The theory here is that poly-tailed losses do not have max-margin behavior
    and thus can work with importance weighting.
    Poly-tailed losses are not defined at v=0 for v^{-\alpha}, and so there are
    several variants that are supported via the [[type]] option
    exp : f(v):= exp(-v+1) for v < 1, 1/v^\alpha otherwise
    logit: f(v):= 1/log(2)log(1+exp(-v+1)) for v < 1, 1/v^\alpha else.
    """

    allowed_types = {"logit"}

    def __init__(self, type: str, alpha: float, reduction: str = "none"):
        super().__init__()
        self.type = type
        assert type in self.allowed_types
        self.alpha = float(alpha)
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= 1
        inv_part = torch.pow(
            margin_vals.abs(), -1 * self.alpha
        )  # prevent exponentiating negative numbers by fractional powers
        if self.type == "logit":
            indicator = margin_vals <= 1
            inv_part = torch.pow(margin_vals.abs(),-1*self.alpha)
            logit_inner = -1 * margin_vals
            logit_part = torch.nn.functional.softplus(logit_inner)/(math.log(1+math.exp(-1)))
            scores = logit_part * indicator + inv_part * (~indicator)
            return scores

    def forward(self, logits, target):
        target_sign = 2 * target - 1
        margin_scores = (logits[:, 1] - logits[:, 0]) * target_sign
        loss_values = self.margin_fn(margin_scores)
        return loss_values

class MCPolynomialLoss(nn.Module):
    """
    Poly-tailed margin based losses that decay as v^{-\alpha} for \alpha > 0.
    The theory here is that poly-tailed losses do not have max-margin behavior
    and thus can work with importance weighting.
    Poly-tailed losses are not defined at v=0 for v^{-\alpha}, and so there are
    several variants that are supported via the [[type]] option
    exp : f(v):= exp(-v+1) for v < 1, 1/v^\alpha otherwise
    logit: f(v):= 1/log(2)log(1+exp(-v+1)) for v < 1, 1/v^\alpha else.
    """

    allowed_types = {"logit"}

    def __init__(self, type: str, alpha: float, reduction: str = "none"):
        super().__init__()
        self.type = type
        assert type in self.allowed_types
        self.alpha = float(alpha)
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= 1
        inv_part = torch.pow(
            margin_vals.abs(), -1 * self.alpha
        )  # prevent exponentiating negative numbers by fractional powers
        if self.type == "logit":
            indicator = margin_vals <= 1
            inv_part = torch.pow(margin_vals.abs(),-1*self.alpha)
            logit_inner = -1 * margin_vals
            logit_part = torch.nn.functional.softplus(logit_inner)/(math.log(1+math.exp(-1)))
            scores = logit_part * indicator + inv_part * (~indicator)
            return scores

    def forward(self, logits, target):
        #target_sign = 2 * target - 1
        #margin_scores = (logits[:, 1] - logits[:, 0]) * target_sign
        #print(logits[:, target])
        #margin_scores = logits.shape[1]*logits[:, target] - torch.sum(logits, dim=1)
        
        tmp_logits = logits.clone()
        tmp_logits[range(target.shape[0]),target] = float("-Inf") 

        margin_scores = logits[range(target.shape[0]), target] - torch.max(tmp_logits, dim=1).values
        #margin_scores = logits[:, target]/torch.sum(logits, dim=1)
        loss_values = self.margin_fn(margin_scores)
        return loss_values


class ResNetMulti(nn.Module):
    def __init__(self, pretrained = False):
        super(ResNetMulti, self).__init__()
        model_resnet18 = models.resnet18(pretrained=pretrained)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

"""
Builds a convolutional neural network on the fashion mnist data set.
Designed to show wandb integration with pytorch.
"""



hyperparameter_defaults = dict(
    batch_size = 100,
    learning_rate = 0.01,
    momentum = 0.9,
    epochs = 500,
    n=2048,
    )

wandb.init(config=hyperparameter_defaults, project="pytorch-cnn-fashion")
config = wandb.config

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=config.channels_one, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=config.channels_one, out_channels=config.channels_two, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(p=config.dropout)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(config.channels_two*4*4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        # Linear function (readout)
        out = self.fc1(out)

        return out

def main():

    n_val=10000

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                  # expand chennel from 1 to 3 to fit 
                                  # ResNet pretrained model
                                  transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                  ])

    train_dataset = dsets.FashionMNIST(root='./data',
                                train=True,
                                transform=transform,
                                download=True
                               )
    
    val_dataset = dsets.FashionMNIST(root='./data',
                                train=True,
                                transform=transform,
                                download=True
                               )

    test_dataset = dsets.FashionMNIST(root='./data',
                                train=False,
                                transform=transform,
                               )
    
    # subset training set
    index_sub = np.random.choice(np.arange(len(train_dataset)), int(config.n+n_val), replace=False)
    ind_train = index_sub[:config.n]
    ind_val = index_sub[config.n:-1]

    # replacing attribute
    train_dataset.data = train_dataset.data[ind_train]
    train_dataset.targets = train_dataset.targets[ind_train]

    print(len(train_dataset))
    val_dataset.data = val_dataset.data[ind_val]
    val_dataset.targets = val_dataset.targets[ind_val]

    label_names = [
        "T-shirt or top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Boot"]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=False)
 
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)

    
    #model = CNNModel()
    model = ResNetMulti()
    
    wandb.watch(model)

    #criterion = nn.CrossEntropyLoss()
    #criterion= PolynomialLoss(type="logit", alpha=1)
    criterion = MCPolynomialLoss(type="logit", alpha=1)

    #optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    iter = 0
    for epoch in range(config.epochs):
        train_acc=[]
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images)
            labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = torch.mean(criterion(outputs, labels))

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            train_acc.append((torch.sum((torch.argmax(outputs, dim=1) == labels))).float())

            iter += 1

        train_accuracy = np.sum(train_acc)/config.n

        # Calculate Val Accuracy
        correct = 0.0
        correct_arr = [0.0] * 10
        total = 0.0
        total_arr = [0.0] * 10

        # Iterate through test dataset
        for images, labels in val_loader:
            images = Variable(images)

            # Forward pass only to get logits/output
            outputs = model(images)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += labels.size(0)
            correct += (predicted == labels).sum()

            for label in range(10):
                correct_arr[label] += (((predicted == labels) & (labels==label)).sum())
                total_arr[label] += (labels == label).sum()

        accuracy = correct / total

        metrics = {'accuracy': accuracy, 'loss': loss, 'train_accuracy': train_accuracy}
        for label in range(10):
            metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

        wandb.log(metrics)

        # Print Loss
        print('Epoch: {0} Loss: {1:.4f} Train_Acc:{3: .4f} Val_Accuracy: {2:.4f}'.format(epoch, loss, accuracy, train_accuracy))
    
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

if __name__ == '__main__':
   main()