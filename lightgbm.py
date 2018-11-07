# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:31:19 2018

@author: YUBO
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn
from pandas import Series,DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from collections import Counter
from sklearn.decomposition import PCA
np.random.seed(0)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV 
import lightgbm as lgb
import time
from scipy.stats import randint as sp_randint
#读取测试集
test_data=pd.read_csv("all_test.csv",header=None)
test_data=test_data.iloc[0:99998,:]
error=pd.read_csv("error.csv",header=None)
test_data=test_data.append(error,ignore_index=True)
test_data1=test_data.iloc[:,1:]
test_data1.columns=list(np.arange(2600))


data_train=pd.read_csv("partTrainData.csv",header=None)
mix_data=data_train.iloc[:,0:2600].append(test_data1,ignore_index=True)
mix_data=DataFrame(mix_data)
scale = StandardScaler()
x = scale.fit_transform(mix_data)#数据标准化
x=DataFrame(x)
pca = PCA(n_components=50)
x_pca_new=pca.fit_transform(x)#在各主成分投影数据，做主成分得分
x_pca_new=DataFrame(x_pca_new)
test11=x_pca_new.iloc[48383:,:]
train11=x_pca_new.iloc[0:48383,:]
train11.loc[:,"class"]=data_train.iloc[:,2600]
train11_train,train11_test=train_test_split(train11,test_size=0.3)#分割训练集测试集

lgbc=lgb.LGBMClassifier(
 task='train',
 max_bin=400,
 num_leaves=11,
 boosting_type='gbdt', 
 learning_rate =0.25,
 n_estimators=296,#296
 max_depth=6,
 min_child_weight=3,
 min_child_samples=20,
 gamma=0,
 subsample=0.7,
 subsample_freq=1,
 colsample_bytree=0.71,
 objective= 'multiclass',
 nthread=4,
 min_split_gain=1e-5,
 num_class=4,
 metric='multi_error',
 random_state=34,
 silent=False
 )
start=time.time()
param1={'learning_rate':[0.25,0.26,0.3,0.31],'n_estimators':sp_randint(200,300)}
gsearch1=RandomizedSearchCV(lgbc,param1,n_iter=100,scoring='f1_macro',cv=3,n_jobs=-1)
gsearch1.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
end=time.time()
end-start


q=gsearch1.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q)
score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q, pos_label=list(set(train11_test.iloc[:,50].values)),average =None)
np.mean(score_test)

start = time.time()
param2={'max_depth':sp_randint(1,10),
 'min_child_weight':sp_randint(1,10)}
gsearch2=RandomizedSearchCV(lgbc,param2,n_iter=80,scoring='f1_macro',cv=5,n_jobs=-1)
gsearch2.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
end=time.time()
lg=end-start
q=gsearch2.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q)
score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q, pos_label=list(set(train11_test.iloc[:,50].values)),average =None)
np.mean(score_test)

start = time.time()
param3 = {
 'subsample':[i/100 for i in range(60,70)],
 'colsample_bytree':[i/100 for i in range(60,70)]
}
gsearch3=RandomizedSearchCV(lgbc,param3,n_iter=80,scoring='f1_macro',cv=3,n_jobs=-1)
gsearch3.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
end=time.time()
lg=end-start
q=gsearch3.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q)
score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q, pos_label=list(set(train11_test.iloc[:,50].values)),average =None)
np.mean(score_test)

start = time.time()
param4 = {'num_leaves':sp_randint(2,100)
 
}
gsearch4=RandomizedSearchCV(lgbc,param4,n_iter=80,scoring='f1_macro',cv=3,n_jobs=-1)
gsearch4.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
end=time.time()
lg=end-start
q=gsearch4.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q)
score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q, pos_label=list(set(train11_test.iloc[:,50].values)),average =None)
np.mean(score_test)


start = time.time()
param4 = {'max_bin':sp_randint(1,500)
 
}
gsearch4=RandomizedSearchCV(lgbc,param4,n_iter=100,scoring='f1_macro',cv=3,n_jobs=-1)
gsearch4.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
end=time.time()
lg=end-start
q=gsearch4.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q)
score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q, pos_label=list(set(train11_test.iloc[:,50].values)),average =None)
np.mean(score_test)


lgbc.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
q=lgbc.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q)
score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q, pos_label=list(set(train11_test.iloc[:,50].values)),average =None)
np.mean(score_test)

p2=lgbc.predict(test11.iloc[:,0:50])
Counter(p2)
test11.loc[:,"class"]=p2

test11.reset_index(drop=True,inplace=True)
test11.loc[:,"label"]=test_data.iloc[:,0]

test11.loc[:,["label","class"]].to_csv("test11c.csv")




