from platform import architecture
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import time
import tqdm as tqdm
from torch.autograd import Variable

import random

import wandb
import sys

import seaborn as sns
sns.set_palette("muted")
import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Linear, Sequential

import wandb
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

class MCPolynomialLoss_max(nn.Module):
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

class MCPolynomialLoss_sum(nn.Module):
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
        
        margin_scores = 10 * logits[range(target.shape[0]), target] - torch.sum(logits, dim=1)
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


class ConvNet(Sequential):
    """Same architecture as Byrd & Lipton 2017 on CIFAR10
    Args:
        output_size: dimensionality of final output, usually the number of classes
    """

    def __init__(self):
        layers = [
            torch.nn.Conv2d(3, 64, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),   #(1, -1),
            torch.nn.Linear(1152, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        ]
        super().__init__(*layers)

    @property
    def linear_output(self):
        return list(self.modules())[0][-1]

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



def get_fmnist_loaders_3channels(n_train, n_val=10000, batch_size_train=None, batch_size=128):
  
  if batch_size_train == None:
    batch_size_train = n_train

  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,)),
                                  # expand chennel from 1 to 3 to fit 
                                  # ResNet pretrained model
                                  transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                  ])

  train_dataset = datasets.FashionMNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True
                              )
  
  val_dataset = datasets.FashionMNIST(root='./data',
                              train=True,
                              transform=transform,
                              download=True
                              )

  test_dataset = datasets.FashionMNIST(root='./data',
                              train=False,
                              transform=transform,
                              )

  # subset training set
  index_sub = np.random.choice(np.arange(len(train_dataset)), int(n_train+n_val), replace=False)
  ind_train = index_sub[:n_train]
  ind_val = index_sub[n_train:-1]

  # replacing attribute
  train_dataset.data = train_dataset.data[ind_train]
  train_dataset.targets = train_dataset.targets[ind_train]

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
                                              batch_size=batch_size_train,
                                              shuffle=True)
  
  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

  return train_loader, val_loader, test_loader

hyperparameter_defaults = dict(
    learning_rate = 0.001,
    epochs = 1000,
    n=64,
    loss_type='ce',
    dataset = 'FashionMNIST',
    architecture = 'ResNet',
    seed = 0,
    momentum=0.9,
    weight_decay=0,
    )

wandb.init(config=hyperparameter_defaults, project="fmnist_multi")
config = wandb.config

torch.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

def main():
  alpha=1
  
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

  train_loader, val_loader, test_loader = get_fmnist_loaders_3channels(config.n, batch_size_train=None, batch_size=128)

  if config.loss_type == 'ce':
    criterion = nn.CrossEntropyLoss(reduction="none")
  elif config.loss_type == 'poly':
    criterion = PolynomialLoss(type="logit", alpha=alpha)
  elif config.loss_type == 'poly-max':
    criterion = MCPolynomialLoss_max(type="logit", alpha=alpha)
  elif config.loss_type == 'poly-sum':
    criterion = MCPolynomialLoss_sum(type="logit", alpha=alpha)

  if config.architecture == 'CNN':
    model = CNNModel()
  elif config.architecture == 'Convnet':
    model = ConvNet()
  elif config.architecture == 'ResNet':
    model = ResNetMulti()
    
  model = model.to(device) 

  wandb.watch(model)
  
  optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
  
  iter=0
  for epoch in range(config['epochs']):
    train_acc=[]
    model.train()

    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images)
        labels = Variable(labels)

        images=images.to(device)
        labels=labels.to(device)

        # if torch.cuda.is_available():
        #   imgs = imgs.cuda()
        #   labels = labels.cuda()

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

    train_accuracy = sum(train_acc)/config.n

    # Calculate Val Accuracy
    # model.eval()

    correct = 0.0
    correct_arr = [0.0] * 10
    total = 0.0
    total_arr = [0.0] * 10

    # Iterate through test dataset
    for images, labels in val_loader:
        images = Variable(images)
        images= images.to(device)
        labels = Variable(labels).to(device)

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
  wandb.finish()


if __name__ == '__main__':
   main()
