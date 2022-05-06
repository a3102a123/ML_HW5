import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.spatial.distance as npd
from scipy.optimize import minimize

# global variable
beta = 5

# Gaussian Process
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
    k = theta[0] * np.exp(-theta[1] * ((a-b) ** 2) / 2)
    # k = npd.euclidean(a,b)
    # k = np.sqrt((a-b)**2)
    return k

def obj_fun(a,b):
    v = lambda x: x[0] * np.exp(-x[1] * ((a-b) ** 2) / 2)
    return v

def log_li(X,Y):
    noise = np.diag(np.ones(len(X))) * (1 / beta)
    C = covar(X,X,[1,1]) + noise
    C_inv = np.linalg.inv(C)
    return -np.log(np.linalg.det(C)) / 2 - (Y.T @ C_inv @Y) / 2 - (len(Y)/2*np.log(np.pi * 2))

def log_likelihood(X,Y):
    noise = np.diag(np.ones(len(X))) * (1 / beta)
    f = lambda x: -np.log(np.linalg.det(covar(X,X,[x[0],x[1]]) + noise)) / 2 - (Y.T @ np.linalg.inv(covar(X,X,[x[0],x[1]]) + noise) @Y) / 2 - (len(Y)/2*np.log(np.pi * 2))
    return f

def covar(X1,X2,theta):
    global beta
    cov = np.zeros((len(X1),len(X2)))
    for i,x1 in enumerate(X1):
        for j,x2 in enumerate(X2):
            cov[i,j] = kernel(x1,x2,theta)
    return cov

def GP(X,Y,theta):
    noise = np.diag(np.ones(len(X))) * (1 / beta)
    # theta = [1,1]
    C = covar(X,X,theta) + noise
    C_inv = np.linalg.inv(C)
    line_X = np.linspace(-60,60,num=1000)
    line_Y = np.zeros(len(line_X))
    line_var = np.zeros(len(line_X))
    for i,x in enumerate(line_X):
        line_Y[i],line_var[i] = GP_predict(X,Y,C_inv,x,theta)
    line_var = 2*abs(line_var)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.title("Gaussian Process")
    plt.ylim(-5,5)
    plt.fill_between(line_X,line_Y + line_var,line_Y - line_var,color="yellow")
    plt.plot(line_X,line_Y)
    plt.scatter(X,Y)

def GP_predict(X,Y,C_inv,x_new,theta):
    noise = 1 / beta 
    K = covar(X,[x_new],theta)
    mu = K.T @ C_inv @ Y
    cov = (kernel(x_new,x_new,theta)+noise) - K.T @ C_inv @ K
    return mu , cov

### reaa data 
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
print(log_li(X,Y))
theta = optimization(X,Y)
GP(X,Y,theta)
plt.show()