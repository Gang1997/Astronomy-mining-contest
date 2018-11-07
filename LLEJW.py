# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:47:34 2018

@author: YUBO
"""

from sklearn import manifold
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn
from pandas import Series,DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,GradientBoostingClassifier
from collections import Counter
from sklearn.preprocessing import StandardScaler
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
lle= manifold.LocallyLinearEmbedding(n_neighbors =10,n_components=50,method="standard",n_jobs =-1)
x_lle_new=lle.fit_transform(x)#在各主成分投影数据，做主成分得分
x_lle_new=DataFrame(x_lle_new)
test11=x_lle_new.iloc[48383:,:]
train11=x_lle_new.iloc[0:48383,:]
train11.loc[:,"class"]=data_train.iloc[:,2600]
train11_train,train11_test=train_test_split(train11,test_size=0.3)


clf1 = RandomForestClassifier(max_features=19,min_samples_leaf=8,n_estimators=265, n_jobs=-1,class_weight="balanced",criterion="gini",oob_score=True)
clf1.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
q1=clf1.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q1)
score_test1 = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q1, pos_label=list(set(train11_test.iloc[:,50].values)),average =None)
np.mean(score_test1)

