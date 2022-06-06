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
# %matplotlib inline

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

"""## **Building the model**

"""
# print(models.resnet18())
class ResNetFeatrueExtractor18(nn.Module):
    def __init__(self, pretrained = False):
        super(ResNetFeatrueExtractor18, self).__init__()
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

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

class ResClassifier(nn.Module):
    def __init__(self, dropout_p=0.5): #in_features=512
        super(ResClassifier, self).__init__()        
        self.fc = nn.Linear(512, 10)
    def forward(self, x):       
        out = self.fc(x)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)

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
def train_with_both_losses(n_train=256, batch_size=256):
  alpha=1
  lr=0.00001
  max_epochs=1000
  min_epochs=50
  
  transform = transforms.Compose([transforms.ToTensor(),
                                  # expand chennel from 1 to 3 to fit 
                                  # ResNet pretrained model
                                  transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                  ]) 

  # download dataset
  mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
  mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
  print(len(mnist_train), len(mnist_test))

#   idx = (mnist_train.targets==0) | (mnist_train.targets==1)
#   mnist_train.targets = mnist_train.targets[idx]
#   mnist_train.data = mnist_train.data[idx]

#   idx_test = (mnist_test.targets==0) | (mnist_test.targets==1)
#   mnist_test.targets = mnist_test.targets[idx_test]
#   mnist_test.data = mnist_test.data[idx_test]

  # subset training set
  index_sub = np.random.choice(np.arange(len(mnist_train)), int(n_train), replace=True)
  # replacing attribute
  mnist_train.data = mnist_train.data[index_sub]
  mnist_train.targets = mnist_train.targets[index_sub]

  # mnist_train.targets[mnist_train.targets < 5] = 0
  # mnist_train.targets[mnist_train.targets >= 5] = 1
  # mnist_test.targets[mnist_test.targets < 5] = 0
  # mnist_test.targets[mnist_test.targets >= 5] = 1

  train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
    shuffle=True, num_workers=0)
  test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
      shuffle=True, num_workers=0)

  for loss_type in ['poly']:   #['ce', 'poly']:
    if loss_type == 'ce':
      criterion = nn.CrossEntropyLoss(reduction="none")
    elif loss_type == 'poly':
      criterion = PolynomialLoss(type="logit", alpha=alpha)

    config = dict(
      n=n_train,
      loss_type=loss_type,
      learning_rate= lr,
      max_epochs= max_epochs,
      batch_size= batch_size,
      pretrained = False,
      dataset='FashionMNIST',
      architecture='Resnet18'
    )

    wandb.init(project="noisy-fair", entity="sml-eth", config=config)

    netG = ResNetFeatrueExtractor18(pretrained = False)
    netF = ResClassifier()

    if torch.cuda.is_available():
        netG = netG.cuda()
        netF = netF.cuda()

    # setting up optimizer for both feature generator G and classifier F.
    opt_g = optim.SGD(netG.parameters(), lr=lr, weight_decay=0.0005)
    opt_f = optim.SGD(netF.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    opt_f = optim.SGD(netF.parameters(), lr=lr, weight_decay=0.0005)

    # setting up optimizer for both feature generator G and classifier F.
    opt_g = optim.SGD(netG.parameters(), lr=lr, weight_decay=0.00005)
    opt_f = optim.SGD(netF.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    opt_f = optim.SGD(netF.parameters(), lr=lr, weight_decay=0.00005)

    loss_final=1000

    for epoch in range(0, max_epochs):
      n, start = 0, time.time()
      train_l_sum = torch.tensor([0.0], dtype=torch.float32)
      train_acc_sum = torch.tensor([0.0], dtype=torch.float32)
      for i, (imgs, labels) in tqdm.tqdm(enumerate(iter(train_loader))):
          netG.train()
          netF.train()
          imgs = Variable(imgs)
          labels = Variable(labels)
          # train on GPU if possible  
          if torch.cuda.is_available():
              imgs = imgs.cuda()
              labels = labels.cuda()
              train_l_sum = train_l_sum.cuda()
              train_acc_sum = train_acc_sum.cuda()

          opt_g.zero_grad()
          opt_f.zero_grad()

          # extracted feature
          bottleneck = netG(imgs)     
          
          # predicted labels
          label_hat = netF(bottleneck)

          # loss function
          loss= torch.sum(criterion(label_hat, labels))
          loss.backward()
          opt_g.step()
          opt_f.step()
          
          # calcualte training error
          netG.eval()
          netF.eval()
          labels = labels.long()
          train_l_sum += loss.float()
          train_acc_sum += (torch.sum((torch.argmax(label_hat, dim=1) == labels))).float()
          n += labels.shape[0]
      test_acc = test_accuracy(iter(test_loader), netG, netF) 
      print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'\
           % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc, time.time() - start))
      
      wandb.log({"loss": train_l_sum/n,
                       "train_accuracy": train_acc_sum/n,
                       "test_accuracy": test_acc
                       }, step=epoch)
      
      # if torch.abs(loss_final-train_l_sum/n) <= 1e-8 and  epoch>min_epochs:
      #     loss_final = train_l_sum/n 
      #     break
      # else:
      #     loss_final = train_l_sum/n 

      if loss_type=='ce':
        ce_test_final = test_acc
      elif loss_type=='poly':
        poly_test_final = test_acc

      # if train_l_sum/n <= 1e-5 and  epoch>min_epochs:
      #   break


    wandb.finish()

  return ce_test_final, poly_test_final


if __name__ == '__main__':
    train_with_both_losses(n_train=int(sys.argv[1]), batch_size=int(sys.argv[1]))
    #globals()[sys.argv[1]](sys.argv[2])

#train_with_both_losses(n_train=64, batch_size=64)