# -*- coding: utf-8 -*-
"""figure2_left_losses.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uk7S-j0OE3UwhlF0CTCTBimyfPT_dvpI

Compute the Error over Different Imbalance Ratios and IW Schemes
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from scipy.stats import norm
from sklearn import svm
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pickle as pkl

import torch
import numpy as np
import torch.optim as optim
import matplotlib as mpl

import seaborn as sns
sns.set_palette("muted")
import matplotlib.pyplot as plt
import math
# %matplotlib inline

import wandb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


from torch import nn


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

def check_accuracy(loader, model):
    device = "cuda"
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        #print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()
    return float(num_correct)/float(num_samples)*100

def train_linear_model(loss_type, weight_type, dataloader_train, dataloader_test, dataloader_val=None, early_stop=False, n_features=2, tau=1):
  alpha = 1.0
  num_epochs = int(1e5)   #int(1e6)
  lr = 1.0
  device = "cuda"  
  torch.random.manual_seed(1)
    
  model = torch.nn.Linear(n_features, 2)
  model = model.to(device)

  if loss_type == "ce":
      loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
  else:
      loss_fn = PolynomialLoss(type="logit", alpha=alpha)

  optimizer = optim.SGD(model.parameters(), lr=lr)

  max_val_accuracy = 0
  reported_test_accuracy = 0

  for i in range(num_epochs):
      for xs, ys in dataloader_train:
          xs = xs.to(device)
          ys = ys.to(device)

          optimizer.zero_grad()

          logits = model(xs)

          if weight_type == "IW":
              loss_ratio = (ys == 1) + (ys == 0) / tau
          elif weight_type == "IW-3":
              loss_ratio = (ys == 1) + (ys == 0) / (tau**3)
          else:
              loss_ratio = torch.ones(ys.shape[0])
          loss_ratio = loss_ratio.to(device)
          
          loss = torch.mean(loss_fn(logits, ys) * loss_ratio)

          loss.backward()

          optimizer.step()

      if i%100 == 0:      #i % 100000 == 0:
          if early_stop:             
            val_accuracy = check_accuracy(dataloader_val, model)
            if val_accuracy > max_val_accuracy:
              max_val_accuracy = val_accuracy
              reported_test_accuracy = check_accuracy(dataloader_test, model)

          if i%100000 == 0:
            train_accuracy = check_accuracy(dataloader_train, model)
            print(f"Epoch {i}: loss={loss.item()}")
            print(f"Epoch {i}: train_accuracy={train_accuracy}")

  if not early_stop:
    reported_test_accuracy = check_accuracy(dataloader_test, model)

  model = model.cpu()

  print("Test Accuracy: " + str(reported_test_accuracy))

  return reported_test_accuracy

def create_dataset(class_one_train_num=20, class_two_train_num=200, class_one_test_num=30, class_two_test_num=30,  n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0, val=False):
  num_train_samples = class_one_train_num + class_two_train_num
  num_test_samples = class_one_test_num + class_two_test_num
  if val:
    num_samples = num_train_samples + 2*num_test_samples
  else:
    num_samples = num_train_samples + num_test_samples

  X, y = make_classification(n_samples=num_samples ,  n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, class_sep=class_sep)

  samples_one = X[y==0]
  labels_one = y[y==0]

  samples_two = X[y==1]
  labels_two = y[y==1]
  
  print(X.shape)

  if val:
    X_one_train, X_one_test, y_one_train, y_one_test = train_test_split(samples_one, labels_one, test_size=2*class_one_test_num, random_state=42)

    X_two_train, X_two_test, y_two_train, y_two_test = train_test_split(samples_two, labels_two, test_size=2*class_two_test_num, random_state=42)

    X_one_val, X_one_test, y_one_val, y_one_test = train_test_split(X_one_test, y_one_test, test_size=class_one_test_num, random_state=42)
    X_two_val, X_two_test, y_one_val, y_two_test = train_test_split(X_two_test, y_two_test, test_size=class_two_test_num, random_state=42)

  else:
    X_one_train, X_one_test, y_one_train, y_one_test = train_test_split(samples_one, labels_one, test_size=class_one_test_num, random_state=42)

    X_two_train, X_two_test, y_two_train, y_two_test = train_test_split(samples_two, labels_two, test_size=class_two_test_num, random_state=42)

  class_one = torch.Tensor(X_one_train)
  class_two = torch.Tensor(X_two_train)
  x_seq = torch.cat((class_one, class_two), dim=0)
  y_seq = torch.cat(
      (torch.zeros(class_one.shape[0], dtype=torch.long), torch.ones(class_two.shape[0], dtype=torch.long))
  )
  
  dataset_train = torch.utils.data.TensorDataset(x_seq, y_seq)
  dataloader_train = torch.utils.data.DataLoader(
      dataset=dataset_train, batch_size=num_train_samples, shuffle=True
  )


  class_one_test = torch.Tensor(X_one_test)
  class_two_test = torch.Tensor(X_two_test)
  x_seq_test = torch.cat((class_one_test, class_two_test), dim=0)
  y_seq_test = torch.cat(
      (torch.zeros(class_one_test.shape[0], dtype=torch.long), torch.ones(class_two_test.shape[0], dtype=torch.long))
  )
  
  dataset_test = torch.utils.data.TensorDataset(x_seq_test, y_seq_test)
  dataloader_test = torch.utils.data.DataLoader(
      dataset=dataset_test, batch_size=int(1e3), shuffle=True
  )  

  if val:
    class_one_val = torch.Tensor(X_one_val)
    class_two_val = torch.Tensor(X_two_val)
    x_seq_val = torch.cat((class_one_val, class_two_val), dim=0)
    y_seq_val = torch.cat(
        (torch.zeros(class_one_val.shape[0], dtype=torch.long), torch.ones(class_two_val.shape[0], dtype=torch.long))
    )
    
    dataset_val = torch.utils.data.TensorDataset(x_seq_val, y_seq_val)
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=int(1e3), shuffle=True
    )

    return dataloader_train, dataloader_test, dataloader_val

  else:
    return dataloader_train, dataloader_test


def run_fig_2_losses(n_train=100, n_test=int(1e4),  n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0, n_runs=5, early_stop=False):

  #approx_taus = [i for i in range(1,12)]
  approx_taus = [1,2,10]
  a_vals = [0., 1., 3.]

  runs = n_runs   #10

  run_data = []

  for run in range(runs):

    print("RUN {} ========================".format(run))

    n1s = []

    taus = []
    perfs = []
    perf_mm = []

    for t in approx_taus:

      n1 = min(int(np.round(t * n/(1.+t))), n-1)
      n2 = n - n1

      n1, n2 = max(n1, n2), min(n1, n2)

      if n1 in n1s:
        continue
      else:
        n1s.append(n1)

      tau = n1/n2

      taus.append(tau)

      print("tau={}, n1={}".format(tau, n1))

      perfs_tau = []

      dataloader_train, dataloader_test, dataloader_val = create_dataset(n1, n2, int(n_test/2), int(n_test/2),  n_features=p, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, class_sep=class_sep, val=True)

      perf_mm.append(train_linear_model("ce", "NO IW", dataloader_train, dataloader_test, dataloader_val, early_stop=early_stop, n_features=n_features, tau=tau))


      print("MM, perf={}".format(perf_mm[-1]))

      for a in a_vals:

        if a == 0.:
          test_acc = train_linear_model("poly", "NO IW", dataloader_train, dataloader_test, dataloader_val, early_stop=early_stop, n_features=n_features, tau=tau)
        elif a== 1.:
          test_acc = train_linear_model("poly", "IW", dataloader_train, dataloader_test, dataloader_val, early_stop=early_stop, n_features=n_features, tau=tau)
        elif a==3.:
          test_acc =train_linear_model("poly", "IW-3", dataloader_train, dataloader_test, dataloader_val, early_stop=early_stop, n_features=n_features, tau=tau)


        perfs_tau.append(test_acc)

        print("IW={}, perf={}".format(a, perfs_tau[-1]))
      perfs.append(perfs_tau)
      print("======================================")

    run_data.append({"run": run, "taus": taus, "a_vals": a_vals,
                    "perfs": perfs, "perf_mm": perf_mm})
    
  return run_data

"""Plot the Performance"""

def fig2_losses(n_train=100, n_test=int(1e4),  n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0, n_runs=10, early_stop=False):
  data = run_fig_2_losses(n_train=n_train, n_test=n_test,  n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, class_sep=class_sep, n_runs=n_runs)

  config = dict(
      n_train= n_train,
      n_test= n_test,
      n_features= n_features,
      n_informative= n_informative,
      n_redundant= n_redundant,
      n_repeated= n_repeated,
      n_useless = n_features - n_redundant-n_informative -n_repeated,
      n_clusters_per_class= n_clusters_per_class,
      class_sep = class_sep,
      n_runs= n_runs,
      early_stop = early_stop,
    )

  wandb.init(project="noisy-fair", entity="sml-eth", config=config)
  
  #Load the data from the pickle file
  taus = data[0]['taus']    #values of the different imbalance ratios. Each value denotes the a value of |P|/|N|
  w_vals = data[0]['a_vals']
  num_runs = len(data)

  #Extract the test error numbers over the various runs from the data file
  test_c0 = np.zeros((num_runs,len(taus)))
  test_c1 = np.zeros((num_runs,len(taus)))
  test_c2 = np.zeros((num_runs,len(taus)))
  test_cmm = np.zeros((num_runs,len(taus)))

  for i in range(num_runs):
      for j in range(len(taus)):
          test_c0[i,j] = data[i]['perfs'][j][0]
          test_c1[i,j] = data[i]['perfs'][j][1]
          test_c2[i,j] = data[i]['perfs'][j][2]
      test_cmm[i,:] = data[i]['perf_mm']

  #calculate average test errors
  avg_test_c0 = np.mean(test_c0,axis=0)
  avg_test_c1 = np.mean(test_c1,axis=0)
  avg_test_c2 = np.mean(test_c2,axis=0)
  avg_test_cmm = np.mean(test_cmm,axis=0)

  std_c0 = np.std(test_c0,axis=0)
  std_c1 = np.std(test_c1,axis=0)
  std_c2 = np.std(test_c2,axis=0)
  std_cmm = np.std(test_cmm,axis=0)

  for j in range(len(taus)):
    wandb.log({"avg_test_c0": avg_test_c0[j],
               "avg_test_c1": avg_test_c1[j],
               "avg_test_c2": avg_test_c2[j],
               "avg_test_cmm": avg_test_cmm[j],
               "taus": taus[j]
              }, step=j)
    
  plt.figure()
  plt.errorbar(taus, avg_test_c0, yerr=std_c0, label="poly no iw")
  plt.errorbar(taus, avg_test_c1, yerr=std_c1, label="poly iw=tau")
  plt.errorbar(taus, avg_test_c2, yerr=std_c2, label="poly iw=tau^3")
  plt.errorbar(taus, avg_test_cmm, yerr=std_cmm, label="ce")
  plt.xlabel("Imbalance Ratio")
  plt.ylabel("Accuracy")
  plt.legend()
  sns.despine()
  plt.savefig("fig_2_losses.pdf")
    
  wandb.finish()


p = 1000  #10000  #1000000

mu_norm = p**0.251

mu_1 = torch.zeros(p)
mu_1[0] = mu_norm

mu_2 = torch.zeros(p)
mu_2[1] = mu_norm

n = 100

fig2_losses(n_train=n, n_test=int(1e4),  n_features=p, n_informative=p, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep=mu_norm/1.4142, n_runs=1)
