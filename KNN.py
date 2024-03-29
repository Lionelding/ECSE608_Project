###################################################################################
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# try push christine
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "/dataset"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

###################################################################################

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

###################################################################################
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import pandas 


# Split-out validation dataset
from sklearn import model_selection

#df_train = pd.read_csv('E:\Coding\GithubSpace\ECSE608_Project\dataset/train.csv')
#df_test=pd.read_csv('E:\Coding\GithubSpace\ECSE608_Project\dataset/test.csv')
#print df_train.columns



#provides data structures to quickly analyze data
#Read the train dataset

dataset = pandas.read_csv("dataset/train.csv")

# Size of the dataframe
print(dataset.shape)

#Drop the first column 'id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:,1:]



#convert categorical data
for column in ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',
 'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
 'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',
 'CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual',
  'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']:
    
    dataset[column] = dataset[column].astype('category')
    dataset[column] = dataset[column].cat.codes
    
    
#seperate the output column which is Saleprice
#Y is the target column, dataset has the rest
Y = dataset.iloc[:,-1]
# drop by Name
dataset = dataset.drop(['SalePrice'], axis=1)


dataset = dataset.fillna(dataset.mean())


validation_size = 0.20
seed = 5

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(dataset, Y, test_size=validation_size, random_state=seed)


models = []
models.append(('LR', LinearRegression()))
models.append(('RIDGE', Ridge()))
models.append(('LASSO', Lasso()))
models.append(('ELN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('SVM', SVR()))
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
    
    
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# train the model
model = Ridge()
model.fit(X_train, Y_train)

#We now have a fit model, pass in our test data to the fitted model to make predictions with it.
predictions = model.predict(X_validation)

#calculate the error
from sklearn.metrics import r2_score
r2_score(Y_validation,predictions)