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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
from sklearn.decomposition import PCA
np.random.seed(0)
from sklearn.preprocessing import StandardScaler
#读取测试集
test_data=pd.read_csv("all_test.csv",header=None)
test_data=test_data.iloc[0:99998,:]
error=pd.read_csv("error.csv",header=None)
test_data=test_data.append(error,ignore_index=True)
test_data1=test_data.iloc[:,1:]
test_data1.columns=list(np.arange(2600))


data_train=np.load("data_oversample.npy")
data_train=DataFrame(data_train)
mix_data=data_train.iloc[:,0:2600].append(test_data1,ignore_index=True)
mix_data=DataFrame(mix_data)
scale = StandardScaler()
x = scale.fit_transform(mix_data)#数据标准化
x=DataFrame(x)
pca = PCA(n_components=50)
x_pca_new=pca.fit_transform(x)#在各主成分投影数据，做主成分得分
x_pca_new=DataFrame(x_pca_new)
test11=x_pca_new.iloc[173564:,:]
train11=x_pca_new.iloc[0:173564,:]
train11.loc[:,"class"]=data_train.iloc[:,2600]
train11_test,train11_train=train_test_split(train11,test_size=0.3)#分割训练集测试

#训练模型
clf1 = RandomForestClassifier(max_features="log2",min_samples_leaf=7,n_estimators=230, n_jobs=-1,class_weight="balanced",criterion="gini",oob_score=True)
clf1.fit(train11_train.iloc[:,0:50],train11_train.iloc[:,50])
q1=clf1.predict(train11_test.iloc[:,0:50])
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50],q1)
#计算F1得分
score_test1 = sklearn.metrics.f1_score(train11_test.iloc[:,50], q1, pos_label=list(set(train11_test.iloc[:,50])),average = None)
np.mean(score_test1)

p2=clf1.predict(test11.iloc[:,0:50])
test11.loc[:,"class"]=p2
test11.reset_index(drop=True,inplace=True)
test11.loc[:,"label"]=test_data.iloc[:,0]
Counter(p2)
test11.loc[:,["label","class"]].to_csv("test12.csv")

