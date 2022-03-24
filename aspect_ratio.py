# -*- coding: utf-8 -*-
"""aspect_ratio.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nm-QodH7hRYGrzFsha8L6DtxrknPr5LA

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

def loss(v, b, z1s, z2s, n1, n2):
  return (torch.sum(1./(z1s @ v)) + b * torch.sum(1./(z2s @ v))) /(n1+b*n2)

def gen_error(v, return_ips=False):
  v1 = v / torch.norm(v)
  v1 = v1.detach().numpy()
  ip1, ip2 = mu_1 @ v1, mu_2 @ v1
  if return_ips:
    return ip1, ip2
  else:
    return 0.5 * (norm.cdf(-ip1) + norm.cdf(-ip2))

def test_error(w, x_test, y_test):
  w = w / torch.norm(w)
  pred = x_test @ w
  pred[pred>=0] = 0
  pred[pred<0] = 1
  err = (pred.int() != y_test.int()).sum()/float(y_test.size(0))*100
  return err

def create_data(class_one_train_num=20, class_two_train_num=200, class_one_test_num=30, class_two_test_num=30,  n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0):
  num_train_samples = class_one_train_num + class_two_train_num
  num_test_samples = class_one_test_num + class_two_test_num
  num_samples = num_train_samples + num_test_samples
  X, y = make_classification(n_samples=num_samples ,  n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, class_sep=class_sep)

  samples_one = X[y==0]
  labels_one = y[y==0]

  X_one_train, X_one_test, y_one_train, y_one_test = train_test_split(samples_one, labels_one, test_size=samples_one.shape[0]-class_one_train_num, random_state=42)

  samples_two = X[y==1]
  labels_two = y[y==1]

  X_two_train, X_two_test, y_two_train, y_two_test = train_test_split(samples_two, labels_two, test_size=samples_two.shape[0]-class_two_train_num, random_state=42)

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
      dataset=dataset_test, batch_size=num_test_samples, shuffle=True
  )  

  return x_seq, y_seq, x_seq_test, y_seq_test, class_one, -class_two

p = 10000  #1000000


n = 100

def run_fig_2(n_train=100, n_test=int(1e4), n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0, n_runs=5, change_d=True, tau=10):

  approx_ar = [0.02, 0.1, 1., 10., 100.]
  a_vals = [0., 1., 3.]

  runs = n_runs   #10

  run_data = []

  for run in range(runs):

    print("RUN {} ========================".format(run))

    n1s = []

    ars = []
    perfs = []
    perf_mm = []

    for t in approx_ar:

      if change_d:
        n_features = int(n_train*t)
      else:
        n_train = int(n_features/t)


      mu_norm = n_features**0.251

      mu_1 = torch.zeros(n_features)
      mu_1[0] = mu_norm

      mu_2 = torch.zeros(n_features)
      if n_features>1:
        mu_2[1] = mu_norm
      else:
        mu_2 = -mu_1


      n1 = min(int(np.round(tau * n_train/(1.+tau))), n-1)
      n2 = n_train - n1
      n1, n2 = max(n1, n2), min(n1, n2)

      ar = n_train/n_features

      ars.append(ar)

      print("aspect_ratio={}, n1={}".format(ar, n1))

      perfs_ar = []

      # z1s = torch.randn((n1, p)) + mu_1[None, :]
      # z2s = torch.randn((n2, p)) + mu_2[None, :]

      # xs = np.vstack((z1s.numpy(), -z2s.numpy()))
      # ys = [0]*n1 + [1]*n2
      
      # z1s_test = torch.randn((1000, p)) + mu_1[None, :]
      # z2s_test = torch.randn((1000, p)) + mu_2[None, :]

      # x_seq_test = np.vstack((z1s.numpy(), -z2s.numpy()))
      # y_seq_test = [0]*n1 + [1]*n2


      xs, ys, x_seq_test, y_seq_test, z1s, z2s = create_data(n1, n2, int(n_test/2), int(n_test/2),  n_features=n_features, n_informative=n_features, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, class_sep=mu_norm*class_sep)
      z1s.requires_grad = False
      z2s.requires_grad = False

      clf = svm.LinearSVC(loss='hinge', fit_intercept=False)
      clf.fit(xs, ys)
      wmm = -torch.Tensor(clf.coef_.flatten())
      perf_mm.append(test_error(wmm, x_seq_test, y_seq_test))


      print("MM, perf={}".format(perf_mm[-1]))

      for a in a_vals:

        b = tau**a

        w = (mu_1 + mu_2).detach()
        w = (w/torch.norm(w)).detach()
        w.requires_grad = True

        optim = torch.optim.SGD([w], lr=1e-3, momentum=0.9)


        while w.grad is None or torch.norm(w.grad) > 1e-5:
          optim.zero_grad()
          l = loss(w, b, z1s, z2s, n1, n2) + torch.norm(w)**2
          l.backward()
          optim.step()
          
        print(test_error(w, xs, ys))
        perfs_ar.append(test_error(w, x_seq_test, y_seq_test))
        #print(w, b, z1s, z2s, n1, n2 )

        print("w={}, perf={}".format(b, perfs_ar[-1]))
      perfs.append(perfs_ar)
      print("======================================")

    run_data.append({"run": run, "ars": ars, "a_vals": a_vals,
                    "perfs": perfs, "perf_mm": perf_mm})
    
  return run_data

"""Plot the Performance"""

def fig2(n_train=100, n_test=int(1e4),  n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.0, n_runs=10, change_d=True, tau=10):
  data = run_fig_2(n_train=n_train, n_test=n_test,  n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, class_sep=class_sep, n_runs=n_runs, change_d=change_d, tau=tau)

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
      change_d = change_d,
      tau = tau,
    )

  wandb.init(project="noisy-fair", entity="sml-eth", config=config)
  
  #Load the data from the pickle file
  ars = data[0]['ars']    #values of the different imbalance ratios. Each value denotes the a value of |P|/|N|
  w_vals = data[0]['a_vals']
  num_runs = len(data)

  #Extract the test error numbers over the various runs from the data file
  test_c0 = np.zeros((num_runs,len(ars)))
  test_c1 = np.zeros((num_runs,len(ars)))
  test_c2 = np.zeros((num_runs,len(ars)))
  test_cmm = np.zeros((num_runs,len(ars)))

  for i in range(num_runs):
      for j in range(len(ars)):
          test_c0[i,j] = data[i]['perfs'][j][0]
          test_c1[i,j] = data[i]['perfs'][j][1]
          test_c2[i,j] = data[i]['perfs'][j][2]
      test_cmm[i,:] = data[i]['perf_mm']

  #calculate average test errors
  avg_test_c0 = np.mean(test_c0,axis=0)
  avg_test_c1 = np.mean(test_c1,axis=0)
  avg_test_c2 = np.mean(test_c2,axis=0)
  avg_test_cmm = np.mean(test_cmm,axis=0)

  for j in range(len(ars)):
    wandb.log({"avg_test_c0": avg_test_c0[j],
               "avg_test_c1": avg_test_c1[j],
               "avg_test_c2": avg_test_c2[j],
               "avg_test_cmm": avg_test_cmm[j],
               "ars": ars[j]
              }, step=j)
    
  wandb.finish()


fig2(n_train=n, n_test=int(1e4),  n_features=p, n_informative=p, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep=1.2, n_runs=5)

fig2(n_train=n, n_test=int(1e4),  n_features=p, n_informative=p, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, class_sep=1.2, n_runs=5, change_d=False)


