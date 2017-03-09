#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('C:/Users/Lione/Desktop/Machine Learning Project/dataset/train.csv')

## Show all of the features
print len(df_train.columns)
print df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
#sns.plt.show()


#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

sns.distplot(data);
sns.plt.show()