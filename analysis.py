# analysis.py
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

# load clean data
table = np.load('data/clean_data.npy')
np.random.shuffle(table)

continuous = table[:,[2,8,9]]
stda = StandardScaler()
continuous = stda.fit_transform(continuous)
table[:,[2,8,9]] = continuous

# this will make the continuous variables to the last position
enc = OneHotEncoder(categorical_features=
	[True,True,False,True,True,True,True,True,False,False])
enc.fit(table)
preprocessed = enc.transform(table)
print('shape after preprocess:',preprocessed.shape)

# PCA 
from sklearn.decomposition import PCA

# pca = PCA(n_components=0.95) # dimensionality reduction
pca = PCA()
transformed = pca.fit_transform(preprocessed.todense())
print('components_',pca.components_)
print('explained_variance_',pca.explained_variance_)
print('explained_variance_ratio_',pca.explained_variance_ratio_)
print('singular_values_',pca.singular_values_)
print('mean_',pca.mean_)
print('n_components_',pca.n_components_)
print('noise_variance_',pca.noise_variance_)

print('Preprocessed data:', preprocessed.todense()[0:10,:])
inversed = pca.inverse_transform(transformed)[0:10,:]
print('After inverse pca',inversed[0:10])

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,4))
# two dimensions reduction plot
plt.plot(transformed[:,0],transformed[:,1],'.',color = '#000000')
plt.savefig('figure/pca2dim.pdf',format='pdf')

# variance ratio plot
fig = plt.figure(figsize=(5,4))
plt.plot(pca.explained_variance_ratio_,color = '#000000')
plt.ylabel('Variance Ratio')
plt.xlabel('Components')
plt.savefig('figure/pca_ratio.pdf',format='pdf')
