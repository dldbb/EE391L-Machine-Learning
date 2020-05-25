#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#decompress data
from zipfile import ZipFile

file_name = "data_GP.zip"

with ZipFile(file_name, 'r') as zip:
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')


# In[ ]:


#read data
import csv
import numpy as np

def readcsv(name):
    data = []
    with open(name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            data.append(row)
    data = np.array(data)
    return data


# In[ ]:


def cleandata(data):
    data_x = data[:, 11]
    data_c = data[:, 14]
    x = []
    orgx = []
    for i in range(len(data)):
        orgx.append(float(data_x[i]))
        if float(data_c[i]) >= 0:
            x.append(float(data_x[i]))
            
    return np.array(orgx), np.array(x)


# In[ ]:


#Get data from 5 trails all together
names = ["CJdata1.csv", "CJdata2.csv", "CJdata3.csv", "CJdata4.csv", "CJdata5.csv"]
orgx = []
x = [] 
for i in range(5):
    data = readcsv(names[i])
    old, new = cleandata(data)
    orgx.append(old)
    x.append(new)


# In[ ]:


import matplotlib.pyplot as plt

#Cleaned data
for i in range(5):
    time = np.linspace(0, len(x[i]), len(x[i]))
    plt.plot(time, x[i], label='CJ0x_' + str(i + 1))
plt.legend()
plt.title('Cleaned Data Points of CJ_0_x')
plt.show()


# In[ ]:


# Original Data
orgx[2] = orgx[2][0:1029]
for i in range(5):
    time = np.linspace(0, len(orgx[i]), len(orgx[i]))
    plt.plot(time, orgx[i], label='CJ0x_original' + str(i + 1))
plt.legend()
plt.title('Original Data Points of CJ_0_x')
plt.show()


# In[ ]:


import sklearn.gaussian_process as gp

#Build GPR Model
kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

time = np.atleast_2d(np.linspace(0, len(x[0]), len(x[0]))).T
model.fit(time, x[0])


# In[ ]:


#Global Kernel Prediction and Plot

pred, sigma = model.predict(time, return_std = True)

l_global = model.kernel_.k2.get_params()['length_scale']
sigma_f_global = np.sqrt(model.kernel_.k1.get_params()['constant_value'])

print(l_global)
print(sigma_f_global)

plt.figure()
plt.plot(time, x[0], 'r.', markersize = 1, label = 'Observation')
plt.plot(time, pred, 'b-', label = 'Prediction')
plt.fill(np.concatenate([time, time[::-1]]),
         np.concatenate([pred - 1.9600 * sigma,
                         (pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='c', ec='None', label='95% confidence interval')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.title("Prediction with Global Kernel")
plt.legend()
plt.show()

'''
{'k1': 0.484**2,
 'k2': RBF(length_scale=93.9),
 'k1__constant_value': 0.23391002643005002,
 'k1__constant_value_bounds': (0.1, 1000.0),
 'k2__length_scale': 93.85473015670743,
 'k2__length_scale_bounds': (0.001, 1000.0)}
'''


# In[ ]:


# try local kernel

def local_GPR(y, start, end):
    #kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(2.0, (1e-3, 1e3))
    #model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

    X = np.atleast_2d(np.linspace(start, end, end - start, endpoint=False)).T
    model.fit(X, y[start:end])
    
    pre, sigma = model.predict(X, return_std = True)
    l = model.kernel_.k2.length_scale#get_params()['length_scale']
    sigma_f = np.sqrt(model.kernel_.k1.constant_value)#get_params()['constant_value'])
    
    return np.linspace(start, end, end - start, endpoint=False), pre, sigma, l, sigma_f


# In[ ]:


import math

window_size = 200
delt = 10

prediction = []
count = []
L = []
SIGMA = []

for i in range(0, len(x[0])):
    prediction.append(0)
    count.append(0)
i = 0

kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

while i < len(x[0]):
    X, pre, sigma, l, sigma_f = local_GPR(x[0], i, min(i + window_size, len(x[0])))
    for j in range(0, len(X)):
        prediction[int(X[j])] += pre[j]
        count[int(X[j])] += 1
    i += delt
    L.append(math.log10(l))
    #L.append(l)
    SIGMA.append(sigma_f)
    
for i in range(0, len(prediction)):
    if count[i] != 0:
        prediction[i] /= count[i]

len(prediction)


# In[ ]:


plt.figure()
time = np.linspace(0, len(prediction), len(prediction))
plt.plot(time, x[0], 'r.', markersize = 1, label = 'Observation')
plt.plot(time, prediction, 'b-', label = 'Prediction')
#plt.fill(np.concatenate([time, time[::-1]]),
#         np.concatenate([pred - 1.9600 * sigma,
#                         (pred + 1.9600 * sigma)[::-1]]),
#         alpha=.5, fc='c', ec='None', label='95% confidence interval')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.title("Prediction with Local Kernel")
plt.legend()
plt.show()


# In[ ]:


#Compare the prediction of different algorithms
plt.figure()
time = np.linspace(0, len(prediction), len(prediction))
plt.plot(time, x[0], 'r.', markersize = 1, label = 'Observation')
plt.plot(time, prediction, 'b-', label = 'Local Prediction')
plt.plot(time, pred, 'g-', label = 'Global Prediction')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.title("Prediction with DIfferent Kernels")
plt.legend()
plt.show()


# In[ ]:


#Evaluation to the prediction accuracy
from sklearn.metrics import mean_squared_error

mse_global = mean_squared_error(x[0], pred)
mse_local = mean_squared_error(x[0], prediction)
print(mse_global)
print(mse_local)


# In[ ]:


#Plot the parameters
plt.figure()
plt.plot(np.linspace(0, len(L), len(x[0])), x[0], markersize = 1, label = 'Motion')
plt.plot(np.linspace(0, len(L), len(L)), L, label = 'Length(log)')
plt.plot(np.linspace(0, len(L), len(L)), SIGMA, label = 'Sigma')
plt.plot(np.linspace(0, len(L), len(L)), np.repeat(2, len(L)), label = 'Initial length')
plt.plot(np.linspace(0, len(L), len(L)), np.repeat(1, len(L)), label = 'Initial sigma')
plt.plot(np.linspace(0, len(L), len(L)), np.repeat(0.2, len(L)), label = 'Noise')
plt.xlabel("Frame")
plt.ylabel("Value")
plt.title("Parameters of Local Kernel")
plt.legend()
plt.show()


# In[ ]:


#Plot the parameters in 2-D
plt.figure()
origi_L = []
for i in range(len(L)):
    origi_L.append(10**L[i])
plt.plot(origi_L, SIGMA, '.', markersize = 10)
plt.xlabel("L")
plt.ylabel("Sigma")
plt.title("2-D Distribution of Parameters of Local Kernel")

plt.show()


# In[ ]:


from sklearn.cluster import KMeans

# Clustering Algorithm for Kernels

#Decide the number of clusters
kernels = []
for i in range(len(SIGMA)):
    kernels.append([origi_L[i], SIGMA[i]])

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(kernels)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


#Plot the result of clustering and count the amount of each cluster
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(kernels)
y_kmeans = kmeans.predict(kernels)
count = []
for i in range(n_clusters):
    c = 0
    for j in range(len(SIGMA)):
        if y_kmeans[j] == i:
            c += 1
    count.append(c)        
print(count)        
plt.scatter(origi_L, SIGMA, c=y_kmeans, label='Parameters')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Cluster centers')
plt.legend()
plt.xlabel("L")
plt.ylabel("Sigma")
plt.title('Kernel Parameters and 3 Cluster Centers')
plt.show()

