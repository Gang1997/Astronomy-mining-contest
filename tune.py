# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:29:46 2018

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
from sklearn.model_selection import GridSearchCV
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
train11_train,train11_test=train_test_split(train11,test_size=0.3)
#参数寻优
results1 = []

# 决策树个数参数取值
n_estimators_options=list(range(100,400,5))

max_depth_options=list(range(39,55,1))
min_samples_split_options=list(range(2,50,2))
max_leaf_nodes_options=list(range(1,50,4))
randomstate_options=list(range(0,50,2))
min_samples_leaf_options=list(range(1,40,2))

for n_estimators_size in n_estimators_options:
    for max_depth_size in max_depth_options:
        clf = RandomForestClassifier(max_features=19,min_samples_leaf=8,n_estimators=265,max_depth=50 ,n_jobs=-1,criterion="gini",oob_score=True,class_weight="balanced")
        clf.fit(train11_train.iloc[:,0:50],train11_train.iloc[:,50])
        q1=clf.predict(train11_test.iloc[:,0:50])
#计算F1得分
        score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50], q1, pos_label=list(set(train11_test.iloc[:,50])),average = None)
        results1.append((max_depth_size ,np.mean(score_test)))
#输出最佳参数组合
print(max(results1, key=lambda x:x[1]))
results2=[]
max_features_options=list(range(37,51,2))
for max_features_size in max_features_options:
    clf = RandomForestClassifier(max_features=29,min_samples_leaf=8,n_estimators=265,max_depth=50 ,n_jobs=-1,criterion="gini",oob_score=True,class_weight="balanced")
    clf.fit(train11_train.iloc[:,0:50],train11_train.iloc[:,50])
    q1=clf.predict(train11_test.iloc[:,0:50])
#计算F1得分
    score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50], q1, pos_label=list(set(train11_test.iloc[:,50])),average = None)
    results2.append((max_features_size ,np.mean(score_test)))

results3=[]
min_samples_leaf_options=list(range(2,30,2))
for   min_samples_leafs in min_samples_leaf_options:
    clf = RandomForestClassifier(min_samples_leaf=4,max_features=29,n_estimators=265,max_depth=50 ,n_jobs=-1,criterion="gini",oob_score=True,class_weight="balanced")
    clf.fit(train11_train.iloc[:,0:50],train11_train.iloc[:,50])
    q1=clf.predict(train11_test.iloc[:,0:50])
#计算F1得分
    score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50], q1, pos_label=list(set(train11_test.iloc[:,50])),average = None)
    results3.append((min_samples_leafs ,np.mean(score_test)))
    
results1=[]
min_samples_split_options=list(range(2,100,2))
for min_samples_splits in min_samples_split_options:
    clf = RandomForestClassifier(min_samples_leaf=4,min_samples_split=min_samples_splits,max_features=29,
                             n_estimators=265,max_depth=50 ,n_jobs=-1,
                             criterion="gini",oob_score=True,class_weight="balanced",random_state=4,bootstrap=True)
    clf.fit(train11_train.iloc[:,0:50],train11_train.iloc[:,50])
    q1=clf.predict(train11_test.iloc[:,0:50])
#计算F1得分
    score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50], q1, pos_label=list(set(train11_test.iloc[:,50])),average = None)
    results1.append((min_samples_splits ,np.mean(score_test)))
#输出最佳参数组合
print(max(results1, key=lambda x:x[1]))

        