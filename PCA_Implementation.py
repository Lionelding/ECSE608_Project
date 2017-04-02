import numpy as np # linear algebra
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


########################## PCA Implementation ################################
### tells how much variability in the data is captured by the jth principal component 

data = pd.read_csv('dataset/good_noPCA.csv')
labels=data["SalePrice"]
test = pd.read_csv('dataset/test.csv')
data = data.drop("SalePrice", 1)
PCA_num=np.shape(data)[1]


index=list(range(0, np.shape(data)[1]))

pca = PCA(whiten=0)
pca.fit(data)
variance = pd.DataFrame(pca.explained_variance_ratio_)
np.cumsum(pca.explained_variance_ratio_)



fig1 = plt.figure()
ax1 = plt.gca()
ax1.plot(index, pca.explained_variance_ratio_,'o-', color="r", label="Accuracy against Alphas")
ax1.legend(loc="best")
ax1.set_yscale('log')
plt.xlabel('jth largest eigenvalue')
plt.ylabel('Magnitude')
plt.title('Magnitude in Logrithm VS j-th principal component ')
#plt.axis('tight')
fig1.savefig('img/PCA_Magnitude_in_Log.png')



fig2 = plt.figure()
ax2 = plt.gca()
ax2.plot(index, pca.explained_variance_ratio_,'o-', color="r", l]abel="Accuracy against Alphas")
ax2.legend(loc="best")
plt.xlabel('jth largest eigenvalue')
plt.ylabel('Magnitude')
plt.title('Magnitude in Decimal VS j-th principal component ')
#plt.axis('tight')
fig1.savefig('img/PCA_Magnitude_in_Decimal.png')







#pca = PCA(n_components=PCA_num,whiten=True)
#pca = pca.fit(data)
#dataPCA = pca.transform(data)
#
#
#
#
#df = pd.DataFrame(dataPCA)
#df.to_csv("dataset/good_PCA.csv", index=False)
