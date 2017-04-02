############################# Load Libraries ##################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf

#############################    Load Data      ###############################ß
data = pd.read_csv('dataset/good_noPCA.csv')
labels=data["SalePrice"]
#test = pd.read_csv('dataset/test.csv')
data = data.drop("SalePrice", 1)
PCA_num=50


## Fill data with missing value
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
data = imp.fit_transform(data)
    
## Perform PCA 
pca = PCA(whiten=True)
pca.fit(data)
variance = pd.DataFrame(pca.explained_variance_ratio_)
np.cumsum(pca.explained_variance_ratio_)


pca = PCA(n_components=PCA_num,whiten=True)
pca = pca.fit(data)
dataPCA = pca.transform(data)



############################  Neural Network ##################################
# Split traing and test
train = dataPCA

print(np.shape(train))

# Shape the labels
labels_nl = labels
labels_nl = labels_nl.reshape(-1,1)


tf.reset_default_graph()
r2 = tflearn.R2()
net = tflearn.input_data(shape=[None, train.shape[1]])
net = tflearn.fully_connected(net, 30, activation='linear')
net = tflearn.fully_connected(net, 10, activation='linear')
net = tflearn.fully_connected(net, 1, activation='linear')
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.01, decay_step=100)
net = tflearn.regression(net, optimizer=sgd,loss='mean_square',metric=r2)
model = tflearn.DNN(net)


model.fit(train, labels_nl,show_metric=True,validation_set=0.2,shuffle=True,n_epoch=50)

predictions_DNN = model.predict(train)
predictions_DNN = np.exp(predictions_DNN)
predictions_DNN = predictions_DNN.reshape(-1,)


#######################  Measure Error   ######################################

#error=0
#for i in range(0,len(predictions_DNN)):
#    single_error=abs(predictions_DNN[i]-np.exp(labels[i]))
#    error=single_error+error
