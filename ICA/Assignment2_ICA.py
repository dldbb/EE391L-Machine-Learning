#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from IPython.display import Audio
from scipy import stats


# In[2]:


def scaler(M):
    N = []
    max_values = M.max(axis=1)
    min_values = M.min(axis=1)
    for i in range(len(M)):
        N.append((M[i] - min_values[i])/(max_values[i] - min_values[i]))
    return np.array(N)
        


# In[3]:


def plotSignal(M):
    rows = len(M)
    N = scaler(M)
    temp = rows * 100 + 10   
    plt.figure()    
    
    for i in range(rows):
        temp = temp + 1
        plt.subplot(temp)
        plt.plot(N[i])
   


# In[4]:


def comparePlot(M, N):
    rows = len(M)
    O = scaler(M)
    P = scaler(N)
   # order = [0, 2, 1]
   # order = [2, 0, 1]
    order = [1, 2, 0]
    temp = rows * 100 + 10   
    plt.figure()    
    
    for i in range(rows):
        temp = temp + 1
        plt.subplot(temp)
        plt.plot(O[i])
        plt.plot(P[order[i]], color="orange")
        
        


# In[5]:


def one_iteration(X, W, ita):
    n = len(W)
    Y = np.dot(W, X)
    Z = 1.0 / (1.0 + np.exp(-Y))
    
    dW = ita * np.dot(np.identity(n) + np.dot((1 - 2 * Z), np.transpose(Y)), W)
    #dW = np.dot(ita * np.identity(n) + ita * np.dot((1 - 2 * Z), np.transpose(Y)), W)
    return dW


# In[6]:


def printCov(U, Y):
    for i in range(len(U)):
        for j in range(len(Y)):
            a=scale(Y[i,:])
            b=scale(U[j,:])
            result=stats.pearsonr(a, b)
            if result[0]>0.5:
                print("recon"+str(i)+" and origin_"+str(j)+" with cov="+str(result))
            else:
                print("Nah")


# In[7]:


def ICA(m, n, X, ita, iteration, U):
    W = 0.01 * np.random.rand(n, m)
    for i in range(iteration):
        dW = one_iteration(X, W, ita)
        W = W + dW
    Y_estimate = np.dot(W, X)
    #Y_estimate = scaler(np.dot(W, X))
    plotSignal(Y_estimate)
    printCov(U, Y_estimate)
    return Y_estimate


# In[9]:


#Test Case:
test = loadmat('./icaTest.mat')
ita = 0.001

At = test['A']
Ut = test['U']
Xt = np.dot(At, Ut)

m = len(Xt)
n = len(Ut)
t = len(Ut[0])
Wt = 0.1 * np.random.rand(n, m)


# In[10]:


plotSignal(Xt)
#scaler(Xt)


# In[11]:


plotSignal(Ut)


# In[12]:


Answer = ICA(3, 3, Xt, ita, 1000000, Ut)


# In[13]:


comparePlot(Ut, Answer)


# In[14]:

#Sounds:
sounds = loadmat('./sounds.mat')['sounds']#5*44000
#Talking, Saw, Applause, Laughing, Plastic sound
U = []
U.append(sounds[0])
#U.append(sounds[1])
U.append(sounds[2])
U.append(sounds[3])
#U.append(sounds[4])
U = np.array(U)
n = len(U)
m = 3

plotSignal(sounds)


# In[15]:


A = np.random.rand(m, n)
X = np.dot(A, U)

Audio(X[0], rate=11025)


# In[16]:


plotSignal(U)


# In[17]:


A


# In[18]:


plotSignal(X)


# In[ ]:


Y = ICA(m, n, X, ita, 5000, U)


# In[ ]:


comparePlot(U, Y)

