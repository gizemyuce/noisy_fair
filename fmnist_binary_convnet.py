# -*- coding: utf-8 -*-
"""ResNet_FashionMNIST_Pytorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14OTLpirQObxWN_lQfFBxJcIa-SQamRRf

# **ResNet Pytorch implementation for FashionMNIST classification**
First we import the required packages.
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
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
# %matplotlib inline

def binary_fmnist_loaders(n_train, n_val=2000, batch_size_train=None, batch_size=128):
  
  if batch_size_train == None:
    batch_size_train = n_train

  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])

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
  
  idx = (train_dataset.targets==0) | (train_dataset.targets==1)
  train_dataset.targets = train_dataset.targets[idx]
  train_dataset.data = train_dataset.data[idx]

  idx = (val_dataset.targets==0) | (val_dataset.targets==1)
  val_dataset.targets = val_dataset.targets[idx]
  val_dataset.data = val_dataset.data[idx]

  idx_test = (test_dataset.targets==0) | (test_dataset.targets==1)
  test_dataset.targets = test_dataset.targets[idx_test]
  test_dataset.data = test_dataset.data[idx_test]
  

  print(len(train_dataset))
  print(n_train+n_val)
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

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(p=0.5)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32*4*4, 2)

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

class ConvNet(Sequential):
    """Same architecture as Byrd & Lipton 2017 on CIFAR10
    Args:
        output_size: dimensionality of final output, usually the number of classes
    """

    def __init__(self):
        layers = [
            torch.nn.Conv2d(1, 64, 3),
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
            torch.nn.Linear(128, 2),
        ]
        super().__init__(*layers)

    @property
    def linear_output(self):
        return list(self.modules())[0][-1]


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


# calculate test accuracy
def test_accuracy(data_iter, netG, netF):
    """Evaluate testset accuracy of a model."""
    acc_sum,n = 0,0
    for (imgs, labels) in data_iter:
        # send data to the GPU if cuda is availabel
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        netG.eval()
        netF.eval()
        with torch.no_grad():
            labels = labels.long()
            acc_sum += torch.sum((torch.argmax(netF(netG(imgs)), dim=1) == labels)).float()
            n += labels.shape[0]
    return acc_sum.item()/n

"""## **Training using Pre-trained model**"""

# Commented out IPython magic to ensure Python compatibility.
# Commented out IPython magic to ensure Python compatibility.
def train_with_both_losses(n_train=256, batch_size=64):
  alpha = 1

  hyperparameter_defaults = dict(
    batch_size = batch_size,
    learning_rate = 0.005,
    epochs = 500,
    n=n_train,
    dataset ='FashionMNIST-binary',
    architecture = 'CNN',
    )

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

  train_loader, val_loader, test_loader = binary_fmnist_loaders(n_train, batch_size_train=batch_size, batch_size=batch_size)

  for loss_type in ['ce', 'poly']:
    config = hyperparameter_defaults
    config['loss_type'] = loss_type

    if loss_type == 'ce':
      criterion = nn.CrossEntropyLoss(reduction="none")
    elif loss_type == 'poly':
      criterion = PolynomialLoss(type="logit", alpha=alpha)


    wandb.init(project="noisy-fair", entity="sml-eth", config=config)

    #model = ConvNet()
    model =CNNModel()

    wandb.watch(model)

    # if torch.cuda.is_available():
    #     model = model.cuda()
        
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    iter = 0
    for epoch in range(config['epochs']):
        train_acc=[]
        # model.train()
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images)
            labels = Variable(labels)

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

        train_accuracy = np.sum(train_acc)/n_train

        # Calculate Val Accuracy
        # model.eval()

        correct = 0.0
        correct_arr = [0.0] * 10
        total = 0.0
        total_arr = [0.0] * 10

        # Iterate through test dataset
        for images, labels in test_loader:
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
    
    if loss_type=='ce':
      ce_test_final = accuracy
    elif loss_type=='poly':
      poly_test_final = accuracy
    
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    wandb.finish()


    # for epoch in range(0, max_epochs):
    #   n, start = 0, time.time()
    #   train_l_sum = torch.tensor([0.0], dtype=torch.float32)
    #   train_acc_sum = torch.tensor([0.0], dtype=torch.float32)
    #   for i, (imgs, labels) in tqdm.tqdm(enumerate(iter(train_loader))):
    #       netG.train()
    #       netF.train()
    #       imgs = Variable(imgs)
    #       labels = Variable(labels)
    #       # train on GPU if possible  
    #       if torch.cuda.is_available():
    #           imgs = imgs.cuda()
    #           labels = labels.cuda()
    #           train_l_sum = train_l_sum.cuda()
    #           train_acc_sum = train_acc_sum.cuda()

    #       opt_g.zero_grad()
    #       opt_f.zero_grad()

    #       # extracted feature
    #       bottleneck = netG(imgs)     
          
    #       # predicted labels
    #       label_hat = netF(bottleneck)

    #       # loss function
    #       loss= torch.mean(criterion(label_hat, labels))
    #       loss.backward()
    #       opt_g.step()
    #       opt_f.step()
          
    #       # calcualte training error
    #       netG.eval()
    #       netF.eval()
    #       labels = labels.long()
    #       train_l_sum += loss.float()
    #       train_acc_sum += (torch.sum((torch.argmax(label_hat, dim=1) == labels))).float()
    #       n += labels.shape[0]
    #   test_acc = test_accuracy(iter(test_loader), netG, netF) 
    #   print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'\
    #        % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc, time.time() - start))
      
    #   wandb.log({"loss": train_l_sum/n,
    #                    "train_accuracy": train_acc_sum/n,
    #                    "test_accuracy": test_acc
    #                    }, step=epoch)
      
    #   # if torch.abs(loss_final-train_l_sum/n) <= 1e-8 and  epoch>min_epochs:
    #   #     loss_final = train_l_sum/n 
    #   #     break
    #   # else:
    #   #     loss_final = train_l_sum/n 

    #   if loss_type=='ce':
    #     ce_test_final = test_acc
    #   elif loss_type=='poly':
    #     poly_test_final = test_acc

    #   # if train_l_sum/n <= 1e-5 and  epoch>min_epochs:
    #   #   break


    # wandb.finish()

  return ce_test_final, poly_test_final


if __name__ == '__main__':
    train_with_both_losses(n_train=int(sys.argv[1]), batch_size=int(sys.argv[1]))
    #globals()[sys.argv[1]](sys.argv[2])

#train_with_both_losses(n_train=64, batch_size=64)
