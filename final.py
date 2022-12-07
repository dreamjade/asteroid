#!/usr/bin/env python
# coding: utf-8

# In[1]:

import csv
#opening the csv file by specifying
with open('sbdb_asteroids.csv') as csv_file:
    # Creating an object of csv reader
    csv_reader = csv.reader(csv_file, delimiter = ',')
    columns = []
 
    # loop to iterate through the rows of csv
    for row in csv_reader:
        # Write columns
        columns.append(row)
# printing the result
column_names = columns[0]

# In[2]:

import numpy as np
from sklearn.model_selection import KFold
from scipy.spatial import distance

# Define the k-NN algorithm
def knn_mul(X, y, x, k, std_adj = np.array([1,1,1,1]), q=2, r= 2):
    # output
    final_predicted = np.zeros((x.shape[0], y.shape[1]))
    
    # Normalize the data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[1:] = X_std[1:]*std_adj # wighted by changing the std
    X_norm = (X - X_mean) / X_std
    x_norm = (x - X_mean) / X_std
    
    # Compute the Minkowski distances between x and all points in X
    distances = distance.cdist (x_norm, X_norm, 'minkowski', p=q)
    
    # Sort the distances and the corresponding labels
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[np.arange(distances.shape[0])[:, None], sorted_indices]
    ys = np.tile(y, (len(x),1,1))
    sorted_labels = ys[np.arange(ys.shape[0])[:, None], sorted_indices]
    # Take the k-nearest neighbors
    nearest_neighbors = sorted_labels[:,:k]

    # Compute the weights based on the distances
    nearest_distances = sorted_distances[:,:k]
    weights = (np.linalg.norm(nearest_distances, ord=-r, axis=-1)[:, np.newaxis] / nearest_distances)**r

    # Compute the weighted average of the labels
    for i in range(x.shape[0]):
        predicted_label = np.dot(nearest_neighbors[i].T, weights[i])
        # Maximize
        final_predicted[i, np.argmax(predicted_label)] = 1
    
    return final_predicted


# In[3]:

# Spec_B list
paraB = [[i[3], i[5], i[6], i[7], i[8]] for i in columns[1:] if '' not in [i[0], i[3], i[5], i[6], i[7], i[8]]]
dataB = [i[0] for i in columns[1:] if '' not in [i[0], i[3], i[5], i[6], i[7], i[8]]]

#[C,S,X,O]
dataB_number = [[1,0,0,0] if i in ['C','Ch','B','Cb','Cgh','Cg','C:'] else i for i in dataB]
dataB_number = [[0,0.75,0,0] if i in ['S','Sq','Sl','L','Sa','K','Sk','Sr','Q','A','S:','R','Sq:','K:','S(IV)'] else i for i in dataB_number]
dataB_number = [[0,0,1,0] if i in ['X','Xc','Xk','Xe','X:'] else i for i in dataB_number]
dataB_number = [[0,0,0,0.9] if i in ['V','T','Ld','D','O','U','V:'] else i for i in dataB_number]

# Define the training data
X = np.array([[float(j) for j in i] for i in paraB])
y = np.array(dataB_number)

# In[5]:

import emcee
def lnlike(theta):
    std_adj = theta[:4]
    number_mul = theta[4:7]
    k = int(theta[7])
    q = theta[8]
    r = theta[9]
    #[C,S,X,O]
    dataB_number = [[1,0,0,0] if i in ['C','Ch','B','Cb','Cgh','Cg','C:'] else i for i in dataB]
    dataB_number = [[0,number_mul[0],0,0] if i in ['S','Sq','Sl','L','Sa','K','Sk','Sr','Q','A','S:','R','Sq:','K:','S(IV)'] else i for i in dataB_number]
    dataB_number = [[0,0,number_mul[1],0] if i in ['X','Xc','Xk','Xe','X:'] else i for i in dataB_number]
    dataB_number = [[0,0,0,number_mul[2]] if i in ['V','T','Ld','D','O','U','V:'] else i for i in dataB_number]

    # Define the training data
    X = np.array([[float(j) for j in i] for i in paraB])
    y = np.array(dataB_number)

    # Run cross-validation to evaluate the performance of the k-NN algorithm
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = []
    for train_index, test_index in kfold.split(X):
        #print(len(train_index), len(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_test = np.array([[1 if j !=0 else 0 for j in i] for i in y_test]) # normalize y_test
        predicted_labels = knn_mul(X_train, y_train, X_test, k, std_adj,q,r)
        score = np.sum(np.apply_along_axis(np.average,1,(y_test*predicted_labels).T))
        scores.append(score)
    return(100*np.log(np.mean(scores)))
    
# In[6]:

def lnprior(theta):
    if theta[7]<20 or theta[7]>100:
        return -np.inf
    for i in theta[:7]:
        if i<0.5 or i>2:
            return -np.inf
    for i in theta[8:10]:
        if i<0.5 or i>2:
            return -np.inf
    return 0

def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)

# In[7]:

#random position
pos = np.random.rand(1024, 10)*[1.5,1.5,1.5,1.5,1.5,1.5,1.5,80,1.5,1.5]+np.ones((1024, 10))*[0.5,0.5,0.5,0.5,0.5,0.5,0.5,20,0.5,0.5]
pos[0] = np.array([0.5,0.5,0.5,0.5,1,1,1,39,2,2])
nwalkers, ndim = pos.shape
filename = "Best_fit_knn.h5"
knn_backend = emcee.backends.HDFBackend(filename, name="Best_fit_knn")
# Don't forget to clear it in case the file already exists
#mcfost_backend.reset(nwalkers, ndim)

# In[8]:

#run mcmc
from multiprocessing import Pool
import os
os.environ["OMP_NUM_THREADS"] = "16"
steps = 16000
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=knn_backend)
    sampler.run_mcmc(None, steps, progress=True)

# In[9]:

import matplotlib.pyplot as plt
fig, axes = plt.subplots(10, figsize=(10, 30), sharex=True)
samples = sampler.get_chain()
labels = ['std_adj','std_adj','std_adj','std_adj','number_mul','number_mul','number_mul','k','q','r']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")

# Save the full figure...
fig.savefig('full_figure.png', bbox_inches='tight')