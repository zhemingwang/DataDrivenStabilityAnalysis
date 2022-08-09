### Download PARC from http://cse.lab.imtlucca.it/~bemporad/parc/

import scipy.io

import numpy as np
from parc import PARC

from time import process_time


K = 9
separation = 'Softmax'
# separation='Voronoi'
sigma = 1
alpha = 1.0e-5
beta = 1.0e-3
softmax_maxiter = 100000
maxiter = 1000
categorical = [False,False]


mat = scipy.io.loadmat('Ybudget.mat')
mat = mat['Ybudget']

p = 2
L = 2
T = 5
Ybudget = np.transpose(mat)
#print(Ybudget)
N = 200
#for N in range(100,2000,100):
Y = Ybudget[0:N,:]
X_train = []
Y_train = []
for i in range(T-L):
    X_train.extend(Y[:,i*p:(L+i)*p]) 
    Y_train.extend(Y[:,(L+i)*p:(L+i+1)*p])

X_train = np.vstack(X_train)
Y_train = np.vstack(Y_train)


t_start = process_time() 

predictor = PARC(K=K, alpha=alpha, sigma=sigma, separation=separation, maxiter=maxiter,
                    cost_tol=1e-4, min_number=10, fit_on_partition=True,
                    beta=beta, verbose=1)
predictor.fit(X_train, Y_train, categorical,weights=np.ones(p))
Kf = predictor.K  # final number of partitions
delta = predictor.delta  # final assignment of training points to clusters
xbar = predictor.xbar  # centroids of final clusters
mdic = {"group_parc": delta}
t_stop = process_time()

print(delta)
print(Kf)
scipy.io.savemat("group_parc.mat", mdic)

print("Elapsed time:", t_stop, t_start) 
   
print("Elapsed time during the whole program in seconds:",t_stop-t_start) 



