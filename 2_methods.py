###################################################################################
import pandas as pd
import numpy as np

from subprocess import check_output
import sys
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
# Split-out validation dataset
from sklearn import model_selection

###################################################################################
## Uncomment below to debug the version 
#print('numpy: {}'.format(np.__version__))
#print('scipy: {}'.format(scipy.__version__))
#print('Python: {}'.format(sys.version))
#print('pandas: {}'.format(pd.__version__))
#print('sklearn: {}'.format(sklearn.__version__))

###################################################################################
#Read the train dataset

dataset = pd.read_csv("dataset/good.csv")
Y = dataset.iloc[:,-1]
dataset = dataset.drop(['SalePrice'], axis=1)
validation_size = 0.20 # ratio between training and validation 
seed = 5
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(dataset, Y, test_size=validation_size, random_state=seed)

##############################      Compare Different Algorithms    ###########################################
models = []
models.append(('LR', LinearRegression()))
models.append(('RIDGE', Ridge()))
models.append(('LASSO', Lasso()))
models.append(('ELN', ElasticNet(alpha=0.0005,l1_ratio=0.001)))
models.append(('KNN', KNeighborsRegressor(n_neighbors=6, weights='uniform',p=1)))
models.append(('SVM', SVR()))
models.append(('RF', RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=6, min_samples_leaf=2)))
scoring = 'r2'


# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,  cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Plot and Save
fig_results = plt.figure()
fig_results.suptitle('Algorithm Comparison')
ax = fig_results.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
fig_results.savefig('img/results.png')



####################################### ElasticNet Regularization and Optimization ############################

#ELN_alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]
#ELN_l1_ratio=[0.001, .01, .1, .5, 0.7, .9, .99]
#
#ELN_result_alphas=[]
#ELN_results_l1_ratio=[]
#for a in ELN_alphas:
#    kfold = model_selection.KFold(n_splits=5, random_state=seed)
#    cv_results_a = model_selection.cross_val_score(ElasticNet(alpha=a), X_train, Y_train, cv=kfold, scoring=scoring)
#    ELN_result_alphas.append(cv_results_a.mean())
#    msg = "%s: %f (%f)" % (name, cv_results_a.mean(), cv_results_a.std())
#    #print(msg)
#    
#for b in ELN_l1_ratio:
#    kfold = model_selection.KFold(n_splits=5, random_state=seed)
#    cv_results_b = model_selection.cross_val_score(ElasticNet(l1_ratio=b), X_train, Y_train, cv=kfold, scoring=scoring) 
#    ELN_results_l1_ratio.append(cv_results_b.mean())
#    msg = "%s: %f (%f)" % (name, cv_results_b.mean(), cv_results_b.std())
#    #print(msg)
#    
#fig1 = plt.figure()
#ax1 = plt.gca()
#ax1.plot(ELN_alphas, ELN_result_alphas,'o-', color="r", label="Accuracy against Alphas")
#ax1.legend(loc="best")
#ax1.set_xscale('log')
#plt.xlabel('ELN_alphas')
#plt.ylabel('Accuracy')
#plt.title('ElasticNet Accuracy VS alphas')
#plt.axis('tight')
#fig1.savefig('img/ELN_ELN_alphas.png')
#
#fig2 = plt.figure()
#ax2 = plt.gca()
#ax2.plot(ELN_l1_ratio, ELN_results_l1_ratio,'o-', color="b", label="Accuracy against L1_ratio")
#ax2.legend(loc="best")
#ax2.set_xscale('log')
#plt.xlabel('ELN_alphas')
#plt.ylabel('Accuracy')
#plt.title('ElasticNet Accuracy VS L1_ratio')
#plt.axis('tight')
#fig2.savefig('img/ELN_l1_ratio.png')



####################################### KNN Regressor Regularization and Optimization ############################
#neighbor=[1,2,3,4,5,6,7,8,9,10,15,20,25]
#p_value=[1,2]
#
#
#KNN_result_neighbor=[]
#KNN_results_p=[]
#for a in neighbor:
#    kfold = model_selection.KFold(n_splits=5, random_state=seed)
#    cv_results_a = model_selection.cross_val_score(KNeighborsRegressor(n_neighbors=a, weights='uniform',p=1), X_train, Y_train, cv=kfold, scoring=scoring)
#    KNN_result_neighbor.append(cv_results_a.mean())
#    msg = "%s %f: %f (%f)" % ("KNN", a, cv_results_a.mean(), cv_results_a.std())
#    #print(msg)
#    
#for b in p_value:
#    kfold = model_selection.KFold(n_splits=5, random_state=seed)
#    cv_results_b = model_selection.cross_val_score(KNeighborsRegressor(p=b), X_train, Y_train, cv=kfold, scoring=scoring) 
#    KNN_results_p.append(cv_results_b.mean())
#    msg = "%s %f: %f (%f)" % ("KNN", b, cv_results_b.mean(), cv_results_b.std())
#    #print(msg)
#    
#    
#fig3 = plt.figure()
#ax3 = plt.gca()
#ax3.plot(neighbor, KNN_result_neighbor,'o-', color="r", label="Accuracy against neighbor number")
#ax3.legend(loc="best")
#plt.xlabel('neighbor')
#plt.ylabel('Accuracy')
#plt.title('KNN Regressor Accuracy VS neighbor')
#plt.axis('tight')
#fig3.savefig('img/KNN_neighbor.png')
#
#fig4 = plt.figure()
#ax4 = plt.gca()
#ax4.plot(p_value, KNN_results_p,'o-', color="b", label="Accuracy against Power parameter for the Minkowski metric")
#ax4.legend(loc="best")
#plt.xlabel('p')
#plt.ylabel('Accuracy')
#plt.title('KNN Accuracy VS manhattan_distance, and euclidean_distance')
#plt.axis('tight')
#fig4.savefig('img/KNN_p_value.png')

####################################### Random Forest Regressor Regularization and Optimization ############################

#n_estimators=[1,2,3,4,5,6,7,10, 15,20,25,30,40,40,60]
#max_depth=[3,5,7,9,10,11,15,20]
#min_samples_split=[3,4,5,6,10,15,20]
#min_samples_leaf=[1,2,3,4,5,10,15,20]
#
#
#
#RF_result_estimator=[]
#RF_result_max_depth=[]
#RF_result_min_samples_split=[]
#RF_result_min_samples_leaf=[]
#
#for a in n_estimators:
#    kfold = model_selection.KFold(n_splits=5, random_state=seed)
#    cv_results_a = model_selection.cross_val_score(RandomForestRegressor(n_estimators=a), X_train, Y_train, cv=kfold, scoring=scoring)
#    RF_result_estimator.append(cv_results_a.mean())
#    msg = "%s %f: %f (%f)" % ("Random Forest", a, cv_results_a.mean(), cv_results_a.std())
#    #print(msg)
#    
#for b in max_depth:
#    kfold = model_selection.KFold(n_splits=5, random_state=seed)
#    cv_results_b = model_selection.cross_val_score(RandomForestRegressor(max_depth=b), X_train, Y_train, cv=kfold, scoring=scoring) 
#    RF_result_max_depth.append(cv_results_b.mean())
#    msg = "%s %f: %f (%f)" % ("Random Forest", b, cv_results_b.mean(), cv_results_b.std())
#    #print(msg)
#    
#for c in min_samples_split:
#    kfold = model_selection.KFold(n_splits=5, random_state=seed)
#    cv_results_c = model_selection.cross_val_score(RandomForestRegressor(min_samples_split=c), X_train, Y_train, cv=kfold, scoring=scoring)
#    RF_result_min_samples_split.append(cv_results_c.mean())
#    msg = "%s %f: %f (%f)" % ("Random Forest", c, cv_results_c.mean(), cv_results_c.std())
#    #print(msg)
#    
#for d in min_samples_leaf:
#    kfold = model_selection.KFold(n_splits=5, random_state=seed)
#    cv_results_d = model_selection.cross_val_score(RandomForestRegressor(min_samples_leaf=d), X_train, Y_train, cv=kfold, scoring=scoring) 
#    RF_result_min_samples_leaf.append(cv_results_d.mean())
#    msg = "%s %f: %f (%f)" % ("Random Forest", d, cv_results_d.mean(), cv_results_d.std())
#    #print(msg)
#    
#    
#    
#fig5 = plt.figure()
#ax5 = plt.subplot(111)
#ax5.plot(n_estimators, RF_result_estimator,'o-', color="r", label="Accuracy against n_estimators")
#ax5.legend(loc="best")
#plt.xlabel('n_estimators')
#plt.ylabel('Accuracy')
#plt.title('RF Accuracy VS n_estimators')
#plt.axis('tight')
#fig5.savefig('img/RF_n_estimators.png')
#
#fig6 = plt.figure()
#ax6 = plt.subplot(111)
#ax6.plot(max_depth, RF_result_max_depth,'o-', color="b", label="Accuracy against max_depth")
#ax6.legend(loc="best")
#plt.xlabel('max_depth')
#plt.ylabel('Accuracy')
#plt.title('RF Accuracy VS max_depth')
#plt.axis('tight')
#fig6.savefig('img/RF_max_depth.png')
#
#
#fig7 = plt.figure()
#ax7 = plt.subplot(111)
#ax7.plot(min_samples_split, RF_result_min_samples_split,'o-', color="r", label="Accuracy against min_samples_split")
#ax7.legend(loc="best")
#plt.xlabel('min_samples_split')
#plt.ylabel('Accuracy')
#plt.title('RF Accuracy VS min_samples_split')
#plt.axis('tight')
#fig7.savefig('img/RF_min_samples_split.png')
#
#
#fig8= plt.figure()
#ax8= plt.subplot(111)
#ax8.plot(min_samples_leaf, RF_result_min_samples_leaf,'o-', color="b", label="Accuracy against min_samples_leaf")
#ax8.legend(loc="best")
#plt.xlabel('min_samples_leaf')
#plt.ylabel('Accuracy')
#plt.title('RF VS min_samples_leaf')
#plt.axis('tight')
#fig8.savefig('img/RF_min_samples_leaf.png')









# train the model
model = Ridge()
model.fit(X_train, Y_train)

#We now have a fit model, pass in our test data to the fitted model to make predictions with it.
predictions = model.predict(X_validation)

##calculate the error
#from sklearn.metrics import r2_score
#r2_score(Y_validation,predictions)
