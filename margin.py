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

def loss_average_poly(v, b, z1s, z2s, n1, n2):
  return (torch.sum(1./(z1s @ v)) + b * torch.sum(1./(z2s @ v))) /(n1+b*n2)

def loss_average_margin_weighted(v, b, z1s, z2s, n1, n2):
  return - (torch.sum((z1s @ v)) - b * torch.sum((z2s @ v))) /(n1+b*n2)

def loss_average_margin(v, b, z1s, z2s, n1, n2):
  return - (torch.sum((z1s @ v)) -  torch.sum((z2s @ v)))

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

def create_data_sparse(p,n1,n2,n_test, s=1, random_flip_prob=0):
    
    # generating ground truth 
    w_gt = torch.zeros(p)
    w_gt[0:s] = 1/(s ** 0.5)

    # uniformly sampling points and labels
    xs = torch.randn((2*(n1+n2), p))
    ys_noiseless = torch.sign(xs @ w_gt)
    # xs = xs.numpy()
    # ys_noiseless = ys_noiseless.numpy()

    # generating the imbalance in the training data
    samples_one = xs[ys_noiseless==1]
    samples_one = samples_one[0:n1,:]

    samples_two = xs[ys_noiseless==-1]
    samples_two = samples_two[0:n2,:]

    if p==2:
        plt.figure()
        plt.scatter(samples_one[:,0], samples_one[:,1], color='blue')
        plt.scatter(samples_two[:,0], samples_two[:,1], color='red')
        plt.savefig('training_distribution.pdf')

    class_one = torch.Tensor(samples_one)
    class_two = torch.Tensor(samples_two)
    x_seq = torch.cat((class_one, class_two), dim=0)
    y_seq = torch.cat(
        (torch.ones(class_one.shape[0], dtype=torch.long), torch.zeros(class_two.shape[0], dtype=torch.long))
    )

    #add noise to the labels
    if random_flip_prob != 0:
        noise_mask = torch.bernoulli(random_flip_prob*torch.ones_like(y_seq))
        flip_to_0 = torch.logical_and(noise_mask==1, y_seq==1)
        flip_to_1 = torch.logical_and(noise_mask==1, y_seq==0)
        y_seq[flip_to_0] = 0
        y_seq[flip_to_1] = 1

        class_one = x_seq[y_seq==1]
        class_two = x_seq[y_seq==0]

        if p==2:
            plt.figure()
            plt.scatter(class_one[:,0], class_one[:,1], color='blue')
            plt.scatter(class_two[:,0], class_two[:,1], color='red')
            plt.savefig('training_distribution_with_noise.pdf')


    # genrating the test data without imbalanca and label noise
    xs_test = torch.randn((int(2*n_test), int(p)))
    ys_noiseless_test = torch.sign(xs_test @ w_gt)
    ys_noiseless_test[ys_noiseless_test==-1] = 0
    
    return x_seq, 1-y_seq, xs_test, 1-ys_noiseless_test, class_one, -class_two

def margin_classifiers_perf(d=1000,n=100,approx_tau=8, SNR=10, n_test=1e4, s=None, random_flip_prob=0, l1=False):
    
    d_informative = d
    d_redundant = 0
    d_repeated = 0
    n_clusters_per_class = 1

    n1 = min(int(np.round(approx_tau * n/(1.+approx_tau))), n-1)
    n2 = n - n1
    n1, n2 = max(n1, n2), min(n1, n2)

    tau = n1/n2
    
    class_sep = np.sqrt(SNR)

    config = dict(
    n_train= n,
    n_test= n_test,
    n_features= d,
    tau =tau,
    SNR = SNR,
    n_informative= d_informative,
    n_redundant= d_redundant,
    n_repeated= d_repeated,
    n_useless = d - d_redundant-d_informative -d_repeated,
    n_clusters_per_class= n_clusters_per_class,
    class_sep = class_sep,
    s=s,
    label_noise_prob= random_flip_prob,
    )

    wandb.init(project="noisy-fair", entity="sml-eth", config=config)

    if s is None:
        xs, ys, x_seq_test, y_seq_test, z1s, z2s = create_data(n1, n2, int(n_test), int(n_test),  n_features=d, n_informative=d_informative, n_redundant=d_redundant, n_repeated=d_repeated, n_classes=2, n_clusters_per_class=n_clusters_per_class, class_sep=class_sep)
    else:
        xs, ys, x_seq_test, y_seq_test, z1s, z2s = create_data_sparse(d,n1,n2,n_test, s=s, random_flip_prob=random_flip_prob)
        w_gt = torch.zeros(d)
        w_gt[0:s] = 1/(s ** 0.5)

    
    a_vals = [0., 1., 3.]

    # l2 margin
    clf = svm.LinearSVC(loss='hinge', fit_intercept=False)
    clf.fit(xs, ys)
    wmm = -torch.Tensor(clf.coef_.flatten())
    print(wmm)
    perf_train_mm = clf.score(xs, ys)
    err_train_mm = 100*(1.-perf_train_mm)
    err_mm = test_error(wmm, x_seq_test, y_seq_test)

    print("CMM train_err={}, err={}".format( err_train_mm, err_mm))

    errs_avm_poly = []
    errs_train_avm_poly = []

    for a in a_vals:

        b = tau**a

        if s is None:
            #w = (torch.normal(mean=torch.zeros(d), std=torch.ones(d)/np.sqrt(d))).detach()
            w = wmm
        else:
            w = w_gt
        
        w = (w/torch.norm(w)).detach()
        w.requires_grad = True

        lmd = torch.Tensor([1.])
        lmd.requires_grad=True

        #optim = torch.optim.SGD([w], lr=1e-3, momentum=0.9)
        optim = torch.optim.SGD([w], lr=1e-3, momentum=0.9)
        #optim = torch.optim.SGD([w, lmd], lr=1e-4, momentum=0.9)

        #scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(1e3), gamma=0.4)

        steps=0
        while w.grad is None or torch.norm(w.grad) > 1e-5:
          optim.zero_grad()
          l = loss_average_poly(w, b, z1s, z2s, n1, n2) + torch.norm(w)**2
          #l = loss_average_poly(w, b, z1s, z2s, n1, n2) + lmd * (torch.norm(w)**2-1.)**2 
          #print(l.item())
          l.backward()
          optim.step()
          #scheduler.step()
          steps +=1
          if steps%1000 ==0:
            print(l.item())
            print(lmd)
          
        err_train = test_error(w, xs, ys)
        err = test_error(w, x_seq_test, y_seq_test)
        errs_train_avm_poly.append(err_train)
        errs_avm_poly.append(err)
        print(w)
        print("w={}, train_err={}, err={}".format(b, err_train, err))

    wandb.log({"err_test_c0": errs_avm_poly[0],
               "err_test_c1": errs_avm_poly[1],
               "err_test_c2": errs_avm_poly[2],
               "err_test_cmm": err_mm,
               "err_train_c0": errs_train_avm_poly[0],
               "err_train_c1": errs_train_avm_poly[1],
               "err_train_c2": errs_train_avm_poly[2],
               "err_train_cmm": err_train_mm,
              })

    if l1 is False:
        return err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly

    else:



        # l1 margin
        print("====================l1=========================")
        #clf = svm.LinearSVC(penalty='l1', loss='hinge', fit_intercept=False)
        clf = svm.LinearSVC(penalty='l1', fit_intercept=False, dual=False)
        clf.fit(xs, ys)
        wmm = -torch.Tensor(clf.coef_.flatten())
        perf_train_mm_l1 = clf.score(xs, ys)
        err_train_mm_l1 = 100*(1.-perf_train_mm_l1)
        err_mm_l1 = test_error(wmm, x_seq_test, y_seq_test)

        print("CMM train_err={}, err={}".format( err_train_mm_l1, err_mm_l1))

        errs_avm_poly_l1 = []
        errs_train_avm_poly_l1 = []

        for a in a_vals:

            b = tau**a

            if s is None:
                #w = (torch.normal(mean=torch.zeros(d), std=torch.ones(d)/np.sqrt(d))).detach()
                w=wmm

            else:
                w = w_gt
            
            w = (w/torch.norm(w)).detach()
            w.requires_grad = True

            optim = torch.optim.SGD([w], lr=1e-3, momentum=0.9)

            steps=0
            while w.grad is None or torch.norm(w.grad) > 1e-5:
                optim.zero_grad()
                l = loss_average_poly(w, b, z1s, z2s, n1, n2) + torch.norm(w, p=1)
                l.backward()
                optim.step()
                steps += 1
                if steps%1000 ==0:
                    print(l.item())
            
            err_train_l1 = test_error(w, xs, ys)
            err_l1 = test_error(w, x_seq_test, y_seq_test)
            errs_train_avm_poly_l1.append(err_train_l1)
            errs_avm_poly_l1.append(err_l1)

            print("w={}, train_err={}, err={}".format(b, err_train_l1, err_l1))

        wandb.log({"err_test_c0_l1": errs_avm_poly_l1[0],
                "err_test_c1_l1": errs_avm_poly_l1[1],
                "err_test_c2_l1": errs_avm_poly_l1[2],
                "err_test_cmm_l1": err_mm_l1,
                "err_train_c0_l1": errs_train_avm_poly_l1[0],
                "err_train_c1_l1": errs_train_avm_poly_l1[1],
                "err_train_c2_l1": errs_train_avm_poly_l1[2],
                "err_train_cmm_l1": err_train_mm_l1,
                })

    wandb.finish()

    return err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly, err_mm_l1, errs_avm_poly_l1, err_train_mm_l1, errs_train_avm_poly_l1

    

def aspect_ratio_l1(d, n, change_d, n_runs=10):
  
    approx_ar = [0.02, 0.1, 1., 10., 100.]
    approx_ar = np.logspace(-0.5,2.1, num=50)
    approx_ar = np.logspace(-1.5,2.1, num=10)
    print(approx_ar)

    runs = n_runs   #10

    run_data = []

    for run in range(runs):

        print("RUN {} ========================".format(run))

        ars = []
        perfs = []
        perf_mm = []
        perfs_l1 = []
        perf_mm_l1 = []

        for t in approx_ar:
            if change_d:
                d = int(n*t)
            else:
                n = int(d/t)

            ar = n/d

            if ar in ars:
                continue
            else:
                ars.append(ar)

            print("aspect_ratio={}, n={}, d={}".format(ar, n, d))

            err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly, err_mm_l1, errs_avm_poly_l1, err_train_mm_l1, errs_train_avm_poly_l1 = margin_classifiers_perf(d=d,n=n,approx_tau=1, SNR=10, n_test=1e4, s=1, l1=True)
            perf_mm.append(err_mm)
            perfs.append(errs_avm_poly)
            perf_mm_l1.append(err_mm_l1)
            perfs_l1.append(errs_avm_poly_l1)

        run_data.append({"run": run, "ars": ars, "a_vals": [0., 1., 3.],
                        "perfs": perfs, "perf_mm": perf_mm, "perfs_l1": perfs_l1, "perf_mm_l1": perf_mm_l1})

    num_runs = len(run_data)
    data = run_data

    #Extract the test error numbers over the various runs from the data file
    test_c0 = np.zeros((num_runs,len(ars)))
    test_c1 = np.zeros((num_runs,len(ars)))
    test_c2 = np.zeros((num_runs,len(ars)))
    test_cmm = np.zeros((num_runs,len(ars)))
    test_c0_l1 = np.zeros((num_runs,len(ars)))
    test_c1_l1 = np.zeros((num_runs,len(ars)))
    test_c2_l1 = np.zeros((num_runs,len(ars)))
    test_cmm_l1 = np.zeros((num_runs,len(ars)))

    for i in range(num_runs):
        for j in range(len(ars)):
            test_c0[i,j] = data[i]['perfs'][j][0]
            test_c1[i,j] = data[i]['perfs'][j][1]
            test_c2[i,j] = data[i]['perfs'][j][2]
        test_cmm[i,:] = data[i]['perf_mm']
        for j in range(len(ars)):
            test_c0_l1[i,j] = data[i]['perfs_l1'][j][0]
            test_c1_l1[i,j] = data[i]['perfs_l1'][j][1]
            test_c2_l1[i,j] = data[i]['perfs_l1'][j][2]
        test_cmm_l1[i,:] = data[i]['perf_mm_l1']

    #calculate average test errors
    avg_test_c0 = np.mean(test_c0,axis=0)
    avg_test_c1 = np.mean(test_c1,axis=0)
    avg_test_c2 = np.mean(test_c2,axis=0)
    avg_test_cmm = np.mean(test_cmm,axis=0)

    avg_test_c0_l1 = np.mean(test_c0_l1,axis=0)
    avg_test_c1_l1 = np.mean(test_c1_l1,axis=0)
    avg_test_c2_l1 = np.mean(test_c2_l1,axis=0)
    avg_test_cmm_l1 = np.mean(test_cmm_l1,axis=0)

    std_test_c0 = np.std(test_c0,axis=0)
    std_test_c1 = np.std(test_c1,axis=0)
    std_test_c2 = np.std(test_c2,axis=0)
    std_test_cmm = np.std(test_cmm,axis=0)

    std_test_c0_l1 = np.std(test_c0_l1,axis=0)
    std_test_c1_l1 = np.std(test_c1_l1,axis=0)
    std_test_c2_l1 = np.std(test_c2_l1,axis=0)
    std_test_cmm_l1 = np.std(test_cmm_l1,axis=0)
    
    plt.figure()
    plt.errorbar(ars, avg_test_c0, yerr=std_test_c0, label="AM no iw")
    plt.errorbar(ars, avg_test_c1, yerr=std_test_c1, label="AM iw=tau")
    plt.errorbar(ars, avg_test_c2, yerr=std_test_c2, label="AM iw=tau^3")
    plt.errorbar(ars, avg_test_cmm, yerr=std_test_cmm, label="MM")

    plt.errorbar(ars, avg_test_c0_l1, yerr=std_test_c0_l1, label="AM-l1 no iw")
    plt.errorbar(ars, avg_test_c1_l1, yerr=std_test_c1_l1, label="AM-l1 iw=tau")
    plt.errorbar(ars, avg_test_c2_l1, yerr=std_test_c2_l1, label="AM-l1 iw=tau^3")
    plt.errorbar(ars, avg_test_cmm_l1, yerr=std_test_cmm_l1, label="MM-l1")
    
    plt.legend()
    sns.despine()
    plt.savefig("sparse_model.pdf")

    
      


if __name__ == "__main__":
    #aspect_ratio_l1(d=100, n=100, change_d=True, n_runs=1)
    
    margin_classifiers_perf(d=5000,n=200,approx_tau=1, SNR=10, n_test=1e4, s=1, l1=True)
    #margin_classifiers_perf(d=2,n=100,approx_tau=8, SNR=10, n_test=1e4, s=1, random_flip_prob=.02)
    #main()
    