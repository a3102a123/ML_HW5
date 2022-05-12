from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.spatial.distance as npd
from scipy.optimize import minimize
import csv
import os
from libsvm.svmutil import *
import sys
import time

# global variable
beta = 5
kernel_folder = "K_mul"
new_kernel_folder = "K_mul"
img_dir = "image"
is_new_file = True
is_SVM = True

is_test = False
test_num = 50
gamma = 0

# Gaussian Process
### read data
def read_point():
    f = open("data/input.data","r")
    X = []
    Y = []
    for d in f:
        d = d.strip()
        d = d.split(" ")
        X.append(float(d[0]))
        Y.append(float(d[1]))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

### optimization
def optimization(X,Y):
    theta = np.array([1,1])
    args = [X,Y]
    res = minimize(log_likelihood(X,Y),theta,method="SLSQP")
    print(res.success)
    print(res.x)
    return res.x

### kernel function
def kernel(a,b,theta):
    # RBF kernel(Radial basis kernel)
    k = theta[0] * np.exp(-theta[1] * ((a-b) ** 2) / 2)
    return k

def obj_fun(a,b):
    v = lambda x: x[0] * np.exp(-x[1] * ((a-b) ** 2) / 2)
    return v

def log_li(X,Y):
    noise = np.diag(np.ones(len(X))) * (1 / beta)
    C = covar(X,X,[1,1]) + noise
    C_inv = np.linalg.inv(C)
    return -( -np.log(np.linalg.det(C)) / 2 - (Y.T @ C_inv @Y) / 2 - (len(Y)/2*np.log(np.pi * 2)))

def log_likelihood(X,Y):
    noise = np.diag(np.ones(len(X))) * (1 / beta)
    f = lambda x: -( -np.log(np.linalg.det(covar(X,X,[x[0],x[1]]) + noise)) / 2 \
    - (Y.T @ np.linalg.inv(covar(X,X,[x[0],x[1]]) + noise) @ Y) / 2 - (len(Y)/2*np.log(np.pi * 2)))
    return f

# Calculating the covariance matrix
def covar(X1,X2,theta):
    cov = np.zeros((len(X1),len(X2)))
    for i,x1 in enumerate(X1):
        for j,x2 in enumerate(X2):
            cov[i,j] = kernel(x1,x2,theta)
    return cov

def GP(X,Y,theta,title):
    # prepare model parameter
    noise = np.diag(np.ones(len(X))) * (1 / beta)
    C = covar(X,X,theta) + noise
    C_inv = np.linalg.inv(C)

    # draw model prediction
    line_X = np.linspace(-60,60,num=1000)
    line_Y = np.zeros(len(line_X))
    line_var = np.zeros(len(line_X))
    for i,x in enumerate(line_X):
        line_Y[i],line_var[i] = GP_predict(X,Y,C_inv,x,theta)
    line_var = 2*np.sqrt(abs(line_var))
    plt.figure(figsize=(8, 6), dpi=120)
    plt.title(title)
    plt.ylim(-5,5)
    plt.fill_between(line_X,line_Y + line_var,line_Y - line_var,color="yellow")
    plt.plot(line_X,line_Y)
    plt.scatter(X,Y)

# prediction
def GP_predict(X,Y,C_inv,x_new,theta):
    noise = 1 / beta 
    K = covar(X,[x_new],theta)
    mu = K.T @ C_inv @ Y
    cov = (kernel(x_new,x_new,theta)+noise) - K.T @ C_inv @ K
    return mu , cov

# SVM
### read data
def read_num(filename):
    f = open(os.path.join("data",filename), newline='')
    rows = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
    ret = []
    i = 0
    for d in rows:
        ret.append(d)
    ret = np.array(ret)
    return ret

def change_label(Y,num):
    new_Y = (Y == num).astype(float)
    new_Y[Y != num] = -1
    return new_Y

# user defined kernel function , multiply the kernel of RBF & linear
def SVM_kernel(a,b):
    # according to the origin setting of libsvm
    global gamma
    rbf = np.exp(-gamma * np.linalg.norm(a - b) ** 2)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    linear = np.dot(a,b)
    return rbf*linear
    return (rbf + linear) / 2

pre_name = 0
name = "_mul"
# pre-computed kernel
def X_to_kernel(X1,X2):
    global pre_name
    K = np.zeros((len(X1),len(X2) + 1))
    K[:,0] = np.array(range(1,len(X1)+1))
    for i , x1 in enumerate(X1):
        for j , x2 in enumerate(X2):
                K[i,j+1] = SVM_kernel(x1,x2)
    if is_test:
        plt.imshow(K[:,1:])
        # plt.show()
        plt.savefig(os.path.join(img_dir,"Kernel{}{}".format(name,pre_name)))
        pre_name +=1
    return K

def save_SVM_kernel(filename,X1,X2,Y):
    K = X_to_kernel(X1,X2)
    np.save(filename + "X",K)
    np.save(filename + "Y",Y)

def load_SVM_kernel(filename):
    K = np.load(filename + "X" + ".npy")
    Y = np.load(filename + "Y" + ".npy")
    return K,Y

def SVM(X_train,Y_train,X_test,Y_test):
    # KX_train = X_to_kernel(X_train[0:20,:],X_train[0:20,:])
    # KX_test = X_to_kernel(X_test[0:20,:],X_train[0:20,:])
    if is_test:
        sX_train = []
        sY_train = []
        sX_test = []
        sY_test = []
        for i in range(5):
            for j in range(test_num):
                sX_train.append(X_train[i*1000 + j])
                sY_train.append(Y_train[i*1000 + j])
                if test_num%2 == 0:
                    sX_test.append(X_test[i*500 + j])
                    sY_test.append(Y_test[i*500 + j])
        sX_train = np.array(sX_train)
        sY_train = np.array(sY_train)
        sX_test = np.array(sX_test)
        sY_test = np.array(sY_test)

        KX_train,Y_train = X_to_kernel(sX_train,sX_train) , sY_train
        KX_test,Y_test = X_to_kernel(sX_test,sX_train) , sY_test
    else:
        KX_train,Y_train = load_SVM_kernel(os.path.join(kernel_folder,"train"))
        KX_test,Y_test = load_SVM_kernel(os.path.join(kernel_folder,"test"))
    print("Kernel data prepared done")
    kernel_name = ["linear" , "polynomial" , "RBF"]
    color = ["red","blue","yellow"]
    # s : the type of SVM , 0 : C-SVC , 1 : nu-SVC , 2 : one-class SVM
    # t : the type of kernel , 0 : linear , 1 : polynomial , 2 : RBF(radial basis function)
    # --- 4 : precomputed kernel
    # -v k : k-fold cross validation
    
    # user defined kernel multi-class SVM
    prob  = svm_problem(Y_train, KX_train, isKernel=True)
    param = svm_parameter("-s 0 -t 4 -q")
    model_user = svm_train(prob,param)
    r_label , r_acc, r_val = svm_predict(Y_test,KX_test,model_user)
    print(r_acc)
    # return


    # multi-class SVM
    # grid search
    best_m = 0 ; best_c = 0 ; best_acc = 0
    acc_arr = np.zeros((3,10))
    for m in range(3):
        for c in range(1,10):
            acc = svm_train(Y_train,X_train,"-s 0 -t {} -c {} -v 2 -q".format(m,c))
            acc_arr[m,c] = acc
            if best_acc < acc:
                best_m = m
                best_c = c

    print("best kernel : {} , C = {}".format(kernel_name[best_m],best_c))
    model_mul = svm_train(Y_train,X_train,"-s 0 -t {} -c {} -q".format(best_m,best_c))
    r_label , r_acc, r_val = svm_predict(Y_test,X_test,model_mul)
    print(r_acc)

    plt.figure()
    plt.xlabel("C number")
    plt.ylabel("Accuracy")
    plt.xticks(range(0,10,1))
    for m in range(3):
        y = acc_arr[m,:]
        plt.plot(range(1,10),y,'s-',color = color[m], label=kernel_name[m])
    plt.legend(loc = "best")

    # return

    acc_arr = np.zeros(3)
    acc_n_arr = np.zeros((3,5))
    sens_n_arr = np.zeros((3,5))
    spec_n_arr = np.zeros((3,5))
    # using different kernel for multiclass SVM
    for m in range(3):
        print("Using kernel : {}".format(kernel_name[m]))
        # training multi class SVM
        model_mul = svm_train(Y_train,X_train,"-s 0 -t {} -q".format(m))
        r_label , r_acc, r_val = svm_predict(Y_test,X_test,model_mul)
        r_label = np.array(r_label)
        true_table = (r_label == Y_test)
        acc_arr[m] = r_acc[0]
        print(r_acc)
        print("{} / {}".format(true_table.sum(),len(Y_test)))
        # the label of this MNIST begins from 1 to 5
        for i in range(1,6):
            acc,sens,spec = print_SVM_result(r_label,Y_test,i,i-1)
            acc_n_arr[m,i-1] = acc
            sens_n_arr[m,i-1] = sens
            spec_n_arr[m,i-1] = spec
    # draw total accuracy
    plt.figure()
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    for m in range(3):
        y = acc_arr[m]
        plt.bar(kernel_name[m],y,color=color[m])
        plt.text(m,y,'%s'%y,ha='center')
    plt.legend(loc = "best")

    plt.figure()
    plt.xlabel("The label of image")
    plt.ylabel("Accuracy")
    plt.xticks(range(0,5,1))
    for m in range(3):
        y = acc_n_arr[m,:]
        plt.plot(range(5),y,'s-',color = color[m], label=kernel_name[m])
    plt.legend(loc = "best")

    plt.figure()
    plt.xlabel("The label of image")
    plt.ylabel("Sensitivity")
    plt.xticks(range(0,5,1))
    for m in range(3):
        y = sens_n_arr[m,:]
        plt.plot(range(5),y,'s-',color = color[m], label=kernel_name[m])
    plt.legend(loc = "best")

    plt.figure()
    plt.xlabel("The label of image")
    plt.ylabel("Specificity")
    plt.xticks(range(0,5,1))
    for m in range(3):
        y = spec_n_arr[m,:]
        plt.plot(range(5),y,'s-',color = color[m], label=kernel_name[m])
    plt.legend(loc = "best")
    return

    # one-class SVM 
    for i in range(1,2):
        Y_sin_train = change_label(Y_train,i)
        Y_sin_test = change_label(Y_test,i)
        print(Y_sin_test)
        model_sin = svm_train(Y_sin_train,X_train,"-s 2 -t 0 -h 0")
        r_label , r_acc, r_val = svm_predict(Y_sin_test,X_test,model_sin)
        r_label = np.array(r_label)
        print(r_label,Y_sin_test)
        print_SVM_result(r_label,Y_sin_test,1,i-1)
    return

# D1 : prediction , D2 : ground turth
def print_SVM_result(D1,D2,label,num_str):

    D1 = (D1 == label)
    D2 = (D2 == label)
    inv_D1 = np.invert(D1)
    inv_D2 = np.invert(D2)
    TP = np.logical_and(D1,D2).sum()
    FP = np.logical_and(D1,inv_D2).sum()
    FN = np.logical_and(inv_D1,D2).sum()
    TN = np.logical_and(inv_D1,inv_D2).sum()
    
    label = num_str
    print("Confusion Matrix {}:".format(label))
    print("{:15}{:^20}{:^20}".format("","Predict number {} ".format(label),"Predict not number {}".format(label)))
    print("{:15}{:^20}{:^20}".format("Is number {}".format(label),TP,FN))
    print("{:15}{:^20}{:^20}".format("Is not number {}".format(label),FP,TN))
    print("")
    acc = (TP+TN) / len(D1)
    sens = TP / (TP + FN)
    spec = TN / (FP + TN)
    print("Accuracy : {:.5}".format(acc))
    print("Sensitivity (Successfully predict number {}): {:.5}".format(label,sens))
    print("Specificity (Successfully predict not number {}): {:.5}".format(label,spec))
    print("")
    print("----------------------------------------")
    return  acc,sens,spec

def test_kernel(X,Y,classes):
    c_num = len(classes)
    img = np.zeros((c_num,784))
    val = np.zeros((c_num,c_num))
    for i,c in enumerate(classes):
        idx = np.argmax(Y == c)
        img[i] = X[idx]

    for i in range(c_num):
        for j in range(c_num):
            val[i,j] = SVM_kernel(img[i],img[j])
    
    print(val)
    plt.imshow(val)
    plt.show()

### reaa data 
X , Y = read_point()
X_train , Y_train = read_num("X_train.csv") , read_num("Y_train.csv")
X_test , Y_test = read_num("X_test.csv") , read_num("Y_test.csv")
Y_train = Y_train.reshape(len(Y_train))
Y_test = Y_test.reshape(len(Y_test))
classes = np.unique(Y_train)
gamma = 1 / len(X_train[0])
### pre-compute kernel value
strat_time = time.time()
if is_new_file and not os.path.exists(new_kernel_folder):
    os.mkdir(new_kernel_folder)
    save_SVM_kernel(os.path.join(new_kernel_folder , "train") , X_train,X_train,Y_train)
    save_SVM_kernel(os.path.join(new_kernel_folder , "test") , X_test,X_train,Y_test)
else:
    print("Pre-computed Kernel file already exist!")
end_time = time.time()
time_c= end_time - strat_time
min_c = int(time_c / 60)
time_c = time_c - min_c * 60
print('Total time cost : {}m , {:.3f}s'.format(min_c,time_c))
# test_kernel(X_train,Y_train,classes)
# sys.exit()
print(X_train.shape,Y_train.shape ,"\n" , X_test.shape , Y_test.shape)
if is_SVM:
    SVM(X_train,Y_train,X_test,Y_test)
else:
    theta = [1,1]
    GP(X,Y,theta,"Gaussian Process (without optimization)")
    theta = optimization(X,Y)
    GP(X,Y,theta,"Gaussian Process (optimized)")
plt.show()