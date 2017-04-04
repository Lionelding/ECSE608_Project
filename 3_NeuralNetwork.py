############################# Load Libraries ##################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
import sklearn.metrics as metrics

#############################    Load Data      ###############################ß

PCA_allowed=1

#Read the train dataset
## Use the data from PCA
if PCA_allowed:
    dataset = pd.read_csv("dataset/good_PCA36.csv")
    temp=pd.read_csv("dataset/good_noPCA.csv")
    labels = temp.iloc[:,-1]

## Do not use the data from PCA
else: 
    dataset = pd.read_csv("dataset/good_noPCA.csv")
    labels = dataset.iloc[:,-1]
    dataset = dataset.drop(['SalePrice'], axis=1)



## Fill data with missing value
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
dataset = imp.fit_transform(dataset)
    
### Perform PCA 
#pca = PCA(whiten=True)
#pca.fit(data)
#variance = pd.DataFrame(pca.explained_variance_ratio_)
#np.cumsum(pca.explained_variance_ratio_)
#
#
#pca = PCA(n_components=PCA_num,whiten=True)
#pca = pca.fit(data)
#dataPCA = pca.transform(data)



############################  Neural Network ##################################
# Split traing and test
train = dataset

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


model.fit(train, labels_nl,show_metric=True,validation_set=0.2,shuffle=True,n_epoch=5)

predictions_DNN = model.predict(train)
#predictions_DNN = np.exp(predictions_DNN)
#predictions_DNN = predictions_DNN.reshape(-1,)

print ("r2_score: %0.3f" % (metrics.r2_score(labels_nl, predictions_DNN)))
print ("mean_squared_error: %0.3f" % (metrics.mean_squared_error(labels_nl, predictions_DNN)))
print ("mean_absolute_error: %0.3f" % (metrics.mean_absolute_error(labels_nl, predictions_DNN)))
print ("median_absolute_error: %0.3f" % (metrics.median_absolute_error(labels_nl, predictions_DNN)))
#######################  Measure Error   ######################################
