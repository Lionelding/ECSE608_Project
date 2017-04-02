#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from sklearn.decomposition import PCA



########################### PCA Implementation ################################
## This code block is better used seperatedly with different methods
## DO NOT USE IT Directly 

#data = pd.read_csv('dataset/good_noPCA.csv')
#labels=data["SalePrice"]
#test = pd.read_csv('dataset/test.csv')
#data = data.drop("SalePrice", 1)
#PCA_num=36
#
#
#pca = PCA(whiten=True)
#pca.fit(data)
#variance = pd.DataFrame(pca.explained_variance_ratio_)
#np.cumsum(pca.explained_variance_ratio_)
#
#
#pca = PCA(n_components=PCA_num,whiten=True)
#pca = pca.fit(data)
#dataPCA = pca.transform(data)
#df = pd.DataFrame(dataPCA)
#
#df.to_csv("dataset/good_PCA.csv", index=False)
