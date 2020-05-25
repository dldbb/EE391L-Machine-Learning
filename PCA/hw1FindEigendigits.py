#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import data
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import math  

from scipy.linalg import eigh

def hw1FindEigendigits(A):
    #k = len(A[0])
    m = A.mean(1)
    Ap = A - m.reshape((m.shape[0]), 1)
    AT = Ap.transpose()
    cov = np.cov(AT)
    #w, v = LA.eig(cov)
    w, v = eigh(cov)
    
    idx = w.argsort()[::-1] 
    w_sorted = w[idx]
    v_sorted = v[:,idx]
    
    V = np.dot(Ap, v_sorted)
    #Vnorm = normalize(V, axis=0, norm='l2')
    
    eigenvector = V / np.linalg.norm(V, axis=0)
    Vnorm = np.nan_to_num(eigenvector)
    
    
    #norm = np.sqrt(np.sum(Vnorm*Vnorm, axis=0))
     
    #return m, Ap, cov, v, V, Vnorm
    return m, Vnorm, cov


# In[ ]:


#A = np.array([[90,60,90],[90,90,30],[60,60,60], [60,60,90],[30,30,30]]) 
#m, Vnorm, norm= hw1FindEigendigits(A)

l = 100
lt = 1000
T = 20

TrD = data.TrainData() # or TestData
x, y = TrD[0:l] # get first l images

TeD = data.TestData()
xt, yt = TeD[0:lt]

A = np.reshape(x, (l, 784)).transpose()

m, V, eig = hw1FindEigendigits(A)
Vk = V[:, :T]


# In[ ]:


def project(A):#input 784 * l
    A = A - m.reshape((m.shape[0]), 1)
    projected = np.dot(Vk.transpose(), A)
    eigenspace = np.dot(Vk, projected)
    reconstruct = np.reshape(eigenspace.transpose(), (A.shape[1], 28, 28))
    return eigenspace, reconstruct    


# In[ ]:


A_eigenspace, A_reconstruct = project(A)

At = np.reshape(xt, (lt, 784)).transpose()
At_eigenspace, At_reconstruct = project(At)
np.shape(At_eigenspace)


# In[ ]:


def myPlot(A):#input k*28*28 image
    #for i in range(1):
     #   plt.figure()
      #  aimage = A[i]
       # aimage = np.array(aimage, dtype='float')
        #aimage = aimage.reshape((28, 28))
      #  plt.imshow(aimage, cmap='gray')
    #plt.show()
    
    fig=plt.figure()
    columns = 10
    rows = 4
    for i in range(columns*rows):
        aimage = A[i]
        aimage = np.array(aimage, dtype='float')
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(aimage, cmap='gray')
    plt.show()


# In[ ]:


eigenVectors = np.reshape(V.transpose(), (l, 28, 28))
myPlot(eigenVectors)


# In[ ]:


A_reshape = np.reshape(A.transpose(), (l, 28, 28))
myPlot(A_reshape)


# In[ ]:


myPlot(A_reconstruct)


# In[ ]:


def myKNN():
    k = 3
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(A_eigenspace.transpose(), y)
    y_pred = knn.predict(At_eigenspace.transpose())

    #score = metrics.accuracy_score(yt, y_pred)
    s = 0;
    for t in range(len(y_pred)):
        if y_pred[t] == yt[t]:
            s = s + 1
    score = s / len(y_pred)
    return score
    
myKNN()


# In[ ]:


train_size = [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
T_set = [5, 10, 20, 40, 80]
'''
scores = [[0.486, 0.548, 0.593, 0.627, 0.592, 0.629, 0.622, 0.634, 0.641, 0.645, 0.652],
          [0.609, 0.696, 0.741, 0.770, 0.809, 0.820, 0.838, 0.840, 0.843, 0.854, 0.851],
          [0.623, 0.728, 0.798, 0.837, 0.857, 0.882, 0.893, 0.896, 0.906, 0.909, 0.909],
          [0.637, 0.730, 0.791, 0.840, 0.868, 0.880, 0.890, 0.906, 0.914, 0.917, 0.921],
          [0.626, 0.696, 0.778, 0.836, 0.856, 0.877, 0.887, 0.901, 0.900, 0.908, 0.911]]
'''


# In[ ]:


#when k = 3
scores_3=[[0.472, 0.546, 0.617, 0.662, 0.611, 0.670, 0.676, 0.676, 0.679, 0.663, 0.652],
          [0.605, 0.658, 0.741, 0.806, 0.812, 0.828, 0.835, 0.841, 0.850, 0.859, 0.851],
          [0.609, 0.701, 0.789, 0.835, 0.864, 0.891, 0.897, 0.902, 0.903, 0.909, 0.909],
          [0.637, 0.730, 0.791, 0.840, 0.868, 0.880, 0.890, 0.906, 0.914, 0.917, 0.921],
          [0.626, 0.696, 0.778, 0.836, 0.856, 0.877, 0.887, 0.901, 0.900, 0.908, 0.911]]


# In[ ]:


for i in range(len(scores)):
    plt.plot(train_size, scores[len(scores) - i - 1], label=("T = ", T_set[len(scores) -i - 1]))
plt.title('Accuracy Score of 1000 Test Images')
plt.xlabel('Training Size')
plt.ylabel('Accuracy Score')
plt.legend()


# In[ ]:


def plotKNNResult(train_size, T_set):
    for p in range(len(T_set)):
        T = T_set[p]
        
        accuracy = []
        for q in range(len(train_size)):
            x, y = TrD[0:train_size[q]] 
            xt, yt = TeD[0:lt]
            AA = np.reshape(x, (train_size[q], 784)).transpose()
            m, V, cov= hw1FindEigendigits(AA)
            Vk = V[:, :T]
            #A_eigenspace, A_reconstruct = project(A)
            #At = np.reshape(xt, (lt, 784)).transpose()
            #At_eigenspace, At_reconstruct = project(At)
            accuracy.append(myKNN())
            print(accuracy)
        plt.plot(range(len(accuracy)), accuracy)
    plt.legend()

