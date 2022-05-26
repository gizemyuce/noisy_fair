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

from scipy.optimize import linprog
import cvxpy as cp

MAX_STEPS = int(1e5)

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

def margin_classifiers_perf(d=1000,n=100,approx_tau=8, SNR=10, n_test=1e4, s=None, random_flip_prob=0, l1=True, l2=False):
    
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

    z_seq = torch.cat((z1s, z2s), dim=0)
    z_mean = torch.mean(z_seq, dim=0)
    
    a_vals = [0.]

    if l2:
        # l2 margin
        clf = svm.LinearSVC(loss='hinge', fit_intercept=False, C=1e5, max_iter=int(1e8))
        clf.fit(xs, ys)
        wmm = -torch.Tensor(clf.coef_.flatten())
        #print(wmm)
        perf_train_mm = clf.score(xs, ys)
        err_train_mm = 100*(1.-perf_train_mm)
        err_mm = test_error(wmm, x_seq_test, y_seq_test)

        print("CMM train_err={}, err={}".format( err_train_mm, err_mm))

        errs_avm_poly = []
        errs_train_avm_poly = []

        for a in a_vals:

            b = tau**a

            w = (z_mean/torch.norm(z_mean))
            
            err_train = test_error(w, xs, ys)
            err = test_error(w, x_seq_test, y_seq_test)
            errs_train_avm_poly.append(err_train)
            errs_avm_poly.append(err)
            #print(w)
            print("w={}, train_err={}, err={}".format(b, err_train, err))

        wandb.log({"err_test_cavm": errs_avm_poly[0],
                "err_test_cmm": err_mm,
                "err_train_cavm": errs_train_avm_poly[0],
                "err_train_cmm": err_train_mm,
                })
        
        wmm_l2 =  wmm/torch.norm(wmm)
        w_l2= w/torch.norm(w)

    if l1 is False:
        return err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly, None, None, None, None, wmm_l2, w_l2, None, None

    else:
        # l1 margin
        print("====================l1=========================")
        #clf = svm.LinearSVC(penalty='l1', loss='hinge', fit_intercept=False)
        clf = svm.LinearSVC(penalty='l1', fit_intercept=False, dual=False, C=1e5, max_iter=int(1e8))
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

            A_ub = torch.cat( [torch.cat([torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-torch.eye(d), -torch.eye(d)], dim=1), torch.cat([-z_seq, torch.zeros(n, d)], dim=1), torch.cat([torch.zeros(1,d), torch.ones(1,d)], dim=1)], dim=0)
            b_ub = torch.cat([torch.zeros(2*d), torch.zeros(n)-1e-7, torch.ones(1)], dim=0)
            #b_ub[-1] = 1

            # res = linprog (-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0), A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=(None,None))
            # #print(res)
            # print(res['message'])
            # w = res['x'][0:d]
            # w=torch.from_numpy(w).float()

            x = cp.Variable(2*d)
            objective = cp.Minimize(-torch.cat([z_mean,torch.zeros_like(z_mean) ], dim=0) @ x)
            constraints = [A_ub @ x <= b_ub]
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            if d<15:
                print(x.value[0:d])
            else:
                print(x.value[0:15])
            #print(x.value[d:-1])
            #print(A_ub @ x.value )
            w=x.value[0:d]
            w=torch.from_numpy(w).float()


            #print(w)

            err_train_l1 = test_error(w, xs, ys)
            err_l1 = test_error(w, x_seq_test, y_seq_test)
            errs_train_avm_poly_l1.append(err_train_l1)
            errs_avm_poly_l1.append(err_l1)

            print("w={}, train_err={}, err={}".format(b, err_train_l1, err_l1))

        wandb.log({"err_test_cavm_l1": errs_avm_poly_l1[0],
                "err_test_cmm_l1": err_mm_l1,
                "err_train_cavm_l1": errs_train_avm_poly_l1[0],
                "err_train_cmm_l1": err_train_mm_l1,
                })
        
        wmm_l1 = wmm/torch.norm(wmm)
        w_l1 = w/torch.norm(w)

    wandb.finish()

    if l1 and l2:
        return err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly, err_mm_l1, errs_avm_poly_l1, err_train_mm_l1, errs_train_avm_poly_l1, wmm_l2, w_l2, wmm_l1, w_l1
    else:
        return None, None, None, None, err_mm_l1, errs_avm_poly_l1, err_train_mm_l1, errs_train_avm_poly_l1, None, None, wmm_l1, w_l1

    

def aspect_ratio_l1(d, n, change_d, n_runs=10, sigma=0, s=1, l1=True, l2=False):
  
    approx_ar = [0.02, 0.1, 1., 10., 100.]
    approx_ar = np.logspace(-0.5,2.1, num=50)
    approx_ar = np.logspace(-0.7,2.1, num=10)
    approx_ar = np.logspace(-1,3, num=4)
    print(approx_ar)

    runs = n_runs   #10

    run_data = []

    for run in range(runs):

        print("RUN {} ========================".format(run))

        ars = []
        d_values=[]
        perfs = []
        perf_mm = []
        perfs_l1 = []
        perf_mm_l1 = []
        perfs_train = []
        perf_mm_train = []
        perfs_l1_train = []
        perf_mm_l1_train = []

        wmms_l2 = []
        wams_l2 = []
        wmms_l1 = []
        wams_l1  = []

        run_convergence_flag = 0

        for t in approx_ar:
            if change_d:
                d = int(n*t)
            else:
                n = int(d/t)

            if n<=2:
                continue

            ar = d/n

            if ar in ars:
                continue
            else:
                ars.append(ar)
                d_values.append(d)

            print("aspect_ratio={}, n={}, d={}".format(ar, n, d))

            err_mm, errs_avm_poly, err_train_mm, errs_train_avm_poly, err_mm_l1, errs_avm_poly_l1, err_train_mm_l1, errs_train_avm_poly_l1,  wmm_l2, w_l2, wmm_l1, w_l1 = margin_classifiers_perf(d=d,n=n,approx_tau=1, SNR=10, n_test=1e4, s=s, l1=True, random_flip_prob=sigma)
            perf_mm.append(err_mm)
            perfs.append(errs_avm_poly)
            perf_mm_l1.append(err_mm_l1)
            perfs_l1.append(errs_avm_poly_l1)
            perf_mm_train.append(err_train_mm)
            perfs_train.append(errs_train_avm_poly)
            perf_mm_l1_train.append(err_train_mm_l1)
            perfs_l1_train.append(errs_train_avm_poly_l1)
            wmms_l2.append(wmm_l2)
            wams_l2.append(w_l2)
            wmms_l1.append(wmm_l1)
            wams_l1.append(w_l1)

            # if err_train_mm + errs_train_avm_poly + err_train_mm_l1 + errs_train_avm_poly_l1 > 0:
            #     run_convergence_flag +=1

        # if run_convergence_flag > 0:
        #     print("Run discarded")
        # else:
        #     run_data.append({"run": run, "ars": ars, "a_vals": [0., 1., 3.],
        #                     "perfs": perfs, "perf_mm": perf_mm, "perfs_l1": perfs_l1, "perf_mm_l1": perf_mm_l1})

        run_data.append({"run": run, "ars": ars, "a_vals": [0., 1., 3.],
                            "perfs": perfs, "perf_mm": perf_mm, "perfs_l1": perfs_l1, "perf_mm_l1": perf_mm_l1,
                            "perfs_train": perfs_train, "perf_mm_train": perf_mm_train, "perfs_l1_train": perfs_l1_train, "perf_mm_l1_train": perf_mm_l1_train,
                            'wmm_l2': wmms_l2, 'wam_l2': wams_l2, 'wmm_l1': wmms_l1, 'wam_l1': wams_l1})

    num_runs = len(run_data)
    data = run_data

    #Extract the test error numbers over the various runs from the data file
    test_c0 = np.zeros((num_runs,len(ars)))
    test_cmm = np.zeros((num_runs,len(ars)))
    test_c0_l1 = np.zeros((num_runs,len(ars)))
    test_cmm_l1 = np.zeros((num_runs,len(ars)))

    for i in range(num_runs):
        if l2:
            for j in range(len(ars)):
                test_c0[i,j] = data[i]['perfs'][j][0]
            test_cmm[i,:] = data[i]['perf_mm']
        if l1:
            for j in range(len(ars)):
                test_c0_l1[i,j] = data[i]['perfs_l1'][j][0]
            test_cmm_l1[i,:] = data[i]['perf_mm_l1']

    #calculate average test errors
    avg_test_c0 = np.mean(test_c0,axis=0)
    avg_test_cmm = np.mean(test_cmm,axis=0)

    avg_test_c0_l1 = np.mean(test_c0_l1,axis=0)
    avg_test_cmm_l1 = np.mean(test_cmm_l1,axis=0)

    std_test_c0 = np.std(test_c0,axis=0)
    std_test_cmm = np.std(test_cmm,axis=0)

    std_test_c0_l1 = np.std(test_c0_l1,axis=0)
    std_test_cmm_l1 = np.std(test_cmm_l1,axis=0)
    
    plt.figure()
    if l2:
        plt.errorbar(ars, avg_test_c0, yerr=std_test_c0, label="AM-l2")
        plt.errorbar(ars, avg_test_cmm, yerr=std_test_cmm, label="MM-l2")
    if l1:
        plt.errorbar(ars, avg_test_c0_l1, yerr=std_test_c0_l1, label="AM-l1")
        plt.errorbar(ars, avg_test_cmm_l1, yerr=std_test_cmm_l1, label="MM-l1")
    plt.legend()
    plt.xscale("log")
    sns.despine()
    plt.xlabel('Aspect Ratio')
    plt.ylabel("0-1 Error (%)")

    if change_d:
        plt.title("n="+str(n))
    else:
        plt.title("d="+str(d))

    if change_d:
        plt.savefig("figures_interp/sparse_model_n"+ str(n) + "_s"+ str(s)+"_sigma"+str(sigma)+".pdf")
    else:
        plt.savefig("figures_interp/sparse_model_d"+ str(d)+ "_s"+ str(s)+"_sigma"+str(sigma)+".pdf")
    
    
    #Extract the train error numbers over the various runs from the data file
    train_c0 = np.zeros((num_runs,len(ars)))
    train_cmm = np.zeros((num_runs,len(ars)))
    train_c0_l1 = np.zeros((num_runs,len(ars)))
    train_cmm_l1 = np.zeros((num_runs,len(ars)))

    for i in range(num_runs):
        if l2:
            for j in range(len(ars)):
                train_c0[i,j] = data[i]['perfs_train'][j][0]
            train_cmm[i,:] = data[i]['perf_mm_train']
        if l1:
            for j in range(len(ars)):
                train_c0_l1[i,j] = data[i]['perfs_l1_train'][j][0]
            train_cmm_l1[i,:] = data[i]['perf_mm_l1_train']

    #calculate average test errors
    avg_train_c0 = np.mean(train_c0,axis=0)
    avg_train_cmm = np.mean(train_cmm,axis=0)

    avg_train_c0_l1 = np.mean(train_c0_l1,axis=0)
    avg_train_cmm_l1 = np.mean(train_cmm_l1,axis=0)

    std_train_c0 = np.std(train_c0,axis=0)
    std_train_cmm = np.std(train_cmm,axis=0)

    std_train_c0_l1 = np.std(train_c0_l1,axis=0)
    std_train_cmm_l1 = np.std(train_cmm_l1,axis=0)
    
    plt.figure()
    if l2:
        plt.errorbar(ars, avg_train_c0, yerr=std_train_c0, label="AM-l2")
        plt.errorbar(ars, avg_train_cmm, yerr=std_train_cmm, label="MM-l2")
    if l1:
        plt.errorbar(ars, avg_train_c0_l1, yerr=std_train_c0_l1, label="AM-l1")
        plt.errorbar(ars, avg_train_cmm_l1, yerr=std_train_cmm_l1, label="MM-l1")
    plt.legend()
    plt.xscale("log")
    sns.despine()
    plt.xlabel('Aspect Ratio')
    plt.ylabel("0-1 Error (%)")
    plt.title("Training Error")

    if change_d:
        plt.title("n="+str(n))
    else:
        plt.title("d="+str(d))

    if change_d:
        plt.savefig("figures_interp/train_sparse_model_n"+ str(n) + "_s"+ str(s)+"_sigma"+str(sigma)+".pdf")
    else:
        plt.savefig("figures_interp/train_sparse_model_d"+ str(d)+ "_s"+ str(s)+"_sigma"+str(sigma)+".pdf")

    #Estimation Error

    esterr_am_l2=[]
    esterr_mm_l2=[]
    esterr_am_l1=[]
    esterr_mm_l1=[]
    esterr_var_mm_l1=[]
    esterr_var_am_l1 = []
    esterr_var_mm_l2=[]
    esterr_var_am_l2 = []
    
    #Bias Variance

    bias_am_l2=[]
    bias_mm_l2=[]
    var_mm_l2=[]
    var_am_l2 = []
    bias_am_l1=[]
    bias_mm_l1=[]
    var_mm_l1=[]
    var_am_l1 = []

    if l2:
        for j, d_val in enumerate(d_values):
            # generating ground truth 
            w_gt = torch.zeros(d_val)
            w_gt[0:s] = 1/(s ** 0.5)
            w_am_d = torch.zeros((num_runs,d_val))
            for i in range(num_runs):
                w_am_d[i,:] = data[i]['wam_l2'][j]
            w_am_average = torch.mean(w_am_d, dim=0)
            bias_am_l2.append((torch.norm(w_am_average-w_gt)**2).detach().numpy())
            var_am_l2.append((torch.sum(torch.std(w_am_d, dim=0)**2)).detach().numpy())

            error = torch.norm(w_am_d- w_gt, dim=1)**2
            esterr_am_l2.append(torch.mean(error).detach().numpy())
            esterr_var_am_l2.append(torch.std(error).detach().numpy())


        for j, d_val in enumerate(d_values):
            # generating ground truth 
            w_gt = torch.zeros(d_val)
            w_gt[0:s] = 1/(s ** 0.5)
            w_mm_d = torch.zeros((num_runs,d_val))
            for i in range(num_runs):
                w_mm_d[i,:] = data[i]['wmm_l2'][j]
            w_mm_average = torch.mean(w_mm_d, dim=0)
            bias_mm_l2.append((torch.norm(w_mm_average-w_gt)**2).detach().numpy())
            var_mm_l2.append((torch.sum(torch.std(w_mm_d, dim=0)**2)).detach().numpy())

            error = torch.norm(w_mm_d- w_gt, dim=1)**2
            esterr_mm_l2.append(torch.mean(error).detach().numpy())
            esterr_var_mm_l2.append(torch.std(error).detach().numpy())

    if l1:
        for j, d_val in enumerate(d_values):
            # generating ground truth 
            w_gt = torch.zeros(d_val)
            w_gt[0:s] = 1/(s ** 0.5)
            w_am_d = torch.zeros((num_runs,d_val))
            for i in range(num_runs):
                w_am_d[i,:] = data[i]['wam_l1'][j]
            w_am_average = torch.mean(w_am_d, dim=0)
            bias_am_l1.append((torch.norm(w_am_average-w_gt)**2).detach().numpy())
            var_am_l1.append((torch.sum(torch.std(w_am_d, dim=0)**2)).detach().numpy())

            error = torch.norm(w_am_d- w_gt, dim=1)**2
            esterr_am_l1.append(torch.mean(error).detach().numpy())
            esterr_var_am_l1.append(torch.std(error).detach().numpy())


        for j, d_val in enumerate(d_values):
            # generating ground truth 
            w_gt = torch.zeros(d_val)
            w_gt[0:s] = 1/(s ** 0.5)
            w_mm_d = torch.zeros((num_runs,d_val))
            for i in range(num_runs):
                w_mm_d[i,:] = data[i]['wmm_l1'][j]
            w_mm_average = torch.mean(w_mm_d, dim=0)
            bias_mm_l1.append((torch.norm(w_mm_average-w_gt)**2).detach().numpy())
            var_mm_l1.append((torch.sum(torch.std(w_mm_d, dim=0)**2)).detach().numpy())

            error = torch.norm(w_mm_d- w_gt, dim=1)**2
            esterr_mm_l1.append(torch.mean(error).detach().numpy())
            esterr_var_mm_l1.append(torch.std(error).detach().numpy())

    
    plt.figure()
    if l2:
        plt.plot(ars, bias_am_l2, 'b', label='bias-am-l2')
        plt.plot(ars, var_am_l2, 'b:', label='variance-am-l2')
        plt.plot(ars, bias_mm_l2, 'r', label='bias-mm-l2')
        plt.plot(ars, var_mm_l2, 'r:', label='variance-mm-l2')
    if l1:
        plt.plot(ars, bias_am_l1, 'g', label='bias-am-l1')
        plt.plot(ars, var_am_l1, 'g:', label='variance-am-l1')
        plt.plot(ars, bias_mm_l1, 'm', label='bias-mm-l1')
        plt.plot(ars, var_mm_l1, 'm:', label='variance-mm-l1')
    
    plt.legend()
    plt.xscale("log")
    sns.despine()
    plt.xlabel('Aspect Ratio')

    if change_d:
        plt.savefig("figures_interp/bias_variance_n"+ str(n)+ "_s"+ str(s) + "_sigma"+str(sigma)+".pdf")
    else:
        plt.savefig("figures_interp/bias_variance_d"+ str(d)+ "_s"+ str(s) + "_sigma"+str(sigma)+".pdf")

    #Estimation Error

    plt.figure()
    if l2:
        plt.errorbar(ars, esterr_am_l2, yerr=esterr_var_am_l2, label='am_l2')
        plt.errorbar(ars, esterr_mm_l2, yerr=esterr_var_mm_l2, label='mm_l2')
    if l1:
        plt.errorbar(ars, esterr_am_l1, yerr=esterr_var_am_l1, label='am_l1')
        plt.errorbar(ars, esterr_mm_l1, yerr=esterr_var_mm_l1, label='mm_l1')
    
    plt.legend()
    plt.xscale("log")
    sns.despine()
    plt.xlabel('Aspect Ratio (d/n)')

    if change_d:
        plt.savefig("figures_interp/est_err_n"+ str(n)+ "_s"+ str(s) + "_sigma"+str(sigma)+".pdf")
    else:
        plt.savefig("figures_interp/est_err_d"+ str(d)+ "_s"+ str(s) + "_sigma"+str(sigma)+".pdf")
     


if __name__ == "__main__":
    
    #aspect_ratio_l1(d=1000, n=100, change_d=False, n_runs=5, s=5)
    aspect_ratio_l1(d=100, n=10, change_d=True, n_runs=2, s=5)
    
    #margin_classifiers_perf(d=1000,n=1000,approx_tau=1, SNR=10, n_test=1e4, s=1, l1=True)
    #margin_classifiers_perf(d=50,n=10,approx_tau=1, SNR=10, n_test=1e4, s=2, l1=True)
    #margin_classifiers_perf(d=2,n=100,approx_tau=8, SNR=10, n_test=1e4, s=1, random_flip_prob=.02)
    #main()
    