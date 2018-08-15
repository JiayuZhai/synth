import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

table = np.load('ICEM_preprocessed_new.npy')
np.random.shuffle(table)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

continuous = table[:,[2,8,9]]
stda = StandardScaler()
continuous = stda.fit_transform(continuous)
table[:,[2,8,9]] = continuous
# this will make the continuous variables to the last position
enc = OneHotEncoder(categorical_features=
	[True,True,False,True,True,True,True,True,False,False])

enc.fit(table)
# preprocessed = enc.transform(table[np.random.randint(table.shape[0],size=[5000]),:])
preprocessed = enc.transform(table)
print(preprocessed.shape)
# PCA 
from sklearn.decomposition import PCA

# pca = PCA(n_components=0.95)
pca = PCA()
x = pca.fit_transform(preprocessed.todense())
print('components_',pca.components_,pca.components_.shape)
print('explained_variance_',pca.explained_variance_)
print('explained_variance_ratio_',pca.explained_variance_ratio_)
print('singular_values_',pca.singular_values_)
print('mean_',pca.mean_)
print('n_components_',pca.n_components_)
print('noise_variance_',pca.noise_variance_)
print('singular_values_unsorted',np.linalg.eig(pca.get_covariance())[0] )


print(preprocessed.todense()[0:10,:])
xx = pca.inverse_transform(x)[0:10,:]
print('After inverse pca',xx[0:10])
# xx = enc.inverse_transform(xx)
# print('After inverse onehot',xx[0:10])
xx[:,[-3,-2,-1]] = stda.inverse_transform(xx[:,[-3,-2,-1]])
print('After inverse std',xx[0:10])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(5,4))

plt.plot(x[:,0],x[:,1],'.',color = '#000000')
# plt.show()
plt.savefig('pca2_new.pdf',format='pdf')
fig = plt.figure(figsize=(5,4))
# plt.plot([sum(pca.explained_variance_ratio_[0:i]) for i in range(pca.n_components_)])
plt.plot(pca.explained_variance_ratio_,color = '#000000')
plt.ylabel('Variance Ratio')
plt.xlabel('Components')
# plt.show()
plt.savefig('pca2_ratio.pdf',format='pdf')

print('PCA finished')

# # Hierarchical clustering Analysis
# from sklearn.cluster import AgglomerativeClustering

# model = AgglomerativeClustering(n_clusters=5,memory='cache/')
# km_result1 = model.fit_predict(preprocessed.todense())
# model = AgglomerativeClustering(n_clusters=4,memory='cache/')
# km_result2 = model.fit_predict(preprocessed.todense())
# model = AgglomerativeClustering(n_clusters=3,memory='cache/')
# km_result3 = model.fit_predict(preprocessed.todense())
# model = AgglomerativeClustering(n_clusters=2,memory='cache/')
# km_result4 = model.fit_predict(preprocessed.todense())

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(221)
# ax.scatter(x[:,0],x[:,1],c=km_result1)
# ax = fig.add_subplot(222)
# ax.scatter(x[:,0],x[:,1],c=km_result2)
# ax = fig.add_subplot(223)
# ax.scatter(x[:,0],x[:,1],c=km_result3)
# ax = fig.add_subplot(224)
# ax.scatter(x[:,0],x[:,1],c=km_result4)
# plt.savefig('HCA.pdf',format='pdf')
# plt.show()
# print('HCA finished')