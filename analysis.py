import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

table = np.load('ICEM_preprocessed.npy')
np.random.shuffle(table)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

continuous = table[:,[2,6,12]]
stda = StandardScaler()
continuous = stda.fit_transform(continuous)
table[:,[2,6,12]] = continuous
# this will make the continuous variables to the last position
enc = OneHotEncoder(categorical_features=
	[True,True,False,True,True,True,False,True,True,True,True,True,False])

enc.fit(table)
preprocessed = enc.transform(table[np.random.randint(table.shape[0],size=[5000]),:])

# PCA 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x = pca.fit_transform(preprocessed.todense())

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(5,4))

plt.plot(x[:,0],x[:,1],'.')
plt.savefig('pca2.pdf',format='pdf')
plt.show()
print('PCA finished')

# Hierarchical clustering Analysis
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=5,memory='cache/')
km_result1 = model.fit_predict(preprocessed.todense())
model = AgglomerativeClustering(n_clusters=4,memory='cache/')
km_result2 = model.fit_predict(preprocessed.todense())
model = AgglomerativeClustering(n_clusters=3,memory='cache/')
km_result3 = model.fit_predict(preprocessed.todense())
model = AgglomerativeClustering(n_clusters=2,memory='cache/')
km_result4 = model.fit_predict(preprocessed.todense())

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(221)
ax.scatter(x[:,0],x[:,1],c=km_result1)
ax = fig.add_subplot(222)
ax.scatter(x[:,0],x[:,1],c=km_result2)
ax = fig.add_subplot(223)
ax.scatter(x[:,0],x[:,1],c=km_result3)
ax = fig.add_subplot(224)
ax.scatter(x[:,0],x[:,1],c=km_result4)
plt.savefig('HCA.pdf',format='pdf')
plt.show()
print('HCA finished')