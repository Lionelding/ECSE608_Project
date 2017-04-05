############################# Load Libraries ##################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from sklearn import model_selection
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



validation_size = 0.15 # ratio between training and validation 
seed = 5
    
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dataset, labels, test_size=validation_size, random_state=seed)




############################  Neural Network ##################################
# Split traing and test






print(np.shape(X_train))

# Shape the labels
Y_train_nl = Y_train
Y_train_nl = Y_train_nl.reshape(-1,1)


tf.reset_default_graph()
r2 = tflearn.R2()
net = tflearn.input_data(shape=[None, X_train.shape[1]])
net = tflearn.fully_connected(net, 20, activation='linear')
net = tflearn.fully_connected(net, 15, activation='linear')
net = tflearn.fully_connected(net, 1, activation='linear')
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.01, decay_step=100)
net = tflearn.regression(net, optimizer=sgd,loss='mean_square',metric=r2)
model = tflearn.DNN(net)


model.fit(X_train, Y_train_nl,show_metric=True,validation_set=validation_size,shuffle=True,n_epoch=20)



## Cross-Validation Resutls
first_layer=[10, 20, 30, 40, 50]
validatoin_loss=[0.02445, 0.01324, 0.01919, 0.02369, 0.01497]
valid_accuracy=[ 1.0007,.9965, 1.0014, 0.9989, 1.0006]

second_layer=[5,10,15,20,25]
validation_loss2=[0.01790, 0.01361, 0.01327, 0.02672, 0.02352]
valid_accuracy2=[1.0010, 0.9980, 0.9992, 0.9981, 0.9973]



## Save ana plot
plt.rcParams['figure.figsize'] = (20, 10)	

fig1 = plt.figure()
fig1.suptitle('Neural Network Parameter Optimization (K-fold=5)', fontsize=14, fontweight='bold')
ax1 = fig1.add_subplot(121)
ax1.plot(first_layer, validatoin_loss,'o-', color="r")
ax1.legend(loc="best")
plt.xlabel('Neuron number in Hidden Layer 1')
plt.ylabel('Validation Loss')
plt.title('Validation Loss VS Neuron Number')
plt.axis('tight')


ax2 = fig1.add_subplot(122)
ax2.plot(second_layer, validation_loss2,'o-', color="b")
ax2.legend(loc="best")
plt.xlabel('Neuron number in Hidden Layer 2')
plt.ylabel('Validation Loss')
plt.title('Validation Loss VS Neuron Number')
plt.axis('tight')
fig1.savefig('img/NN.png')



predictions_DNN = model.predict(X_test)



print ("r2_score: %0.3f" % (metrics.r2_score(Y_test, predictions_DNN)))
print ("mean_squared_error: %0.3f" % (metrics.mean_squared_error(Y_test, predictions_DNN)))
print ("mean_absolute_error: %0.3f" % (metrics.mean_absolute_error(Y_test, predictions_DNN)))
print ("median_absolute_error: %0.3f" % (metrics.median_absolute_error(Y_test, predictions_DNN)))
#######################  Measure Error   ######################################









