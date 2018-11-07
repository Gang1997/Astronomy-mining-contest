# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 10:49:07 2018

@author: YUBO
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
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
#随机森林预测模型
x=np.load("all_kpca.npy")
kpca_new=DataFrame(x)
kpca_new_test,kpca_new_train=train_test_split(kpca_new,test_size=0.3)#分割训练集测试集


results = []
# 最小叶子结点的参数取值
sample_leaf_options = list(range(1,11,1))
# 决策树个数参数取值
n_estimators_options=list(range(1,302,2))

#参数选择过程
for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        
        clf = RandomForestClassifier(max_features="log2",min_samples_leaf=leaf_size, n_estimators=n_estimators_size,n_jobs=-1,criterion="gini",oob_score=True,class_weight="balanced")
        clf.fit(kpca_new_train.iloc[:,0:50],kpca_new_train.iloc[:,50])
        q1=clf.predict(kpca_new_test.iloc[:,0:50])
#计算F1得分
        score_test = sklearn.metrics.f1_score(kpca_new_test.iloc[:,50], q1, pos_label=list(set(kpca_new_test.iloc[:,50])),average = None)
        results.append((leaf_size,n_estimators_size,np.mean(score_test)))
#输出最佳参数组合
print(max(results, key=lambda x:x[2]))


#使用最佳参数训练随机森林模型
clf1 = RandomForestClassifier(max_features=None,min_samples_leaf=5,n_estimators=273, n_jobs=-1,class_weight="balanced",criterion="gini",oob_score=True)
clf1.fit(kpca_new_train.iloc[:,0:50],kpca_new_train.iloc[:,50])
q1=clf1.predict(kpca_new_test.iloc[:,0:50])
sklearn.metrics.confusion_matrix(kpca_new_test.iloc[:,50],q1)
#计算F1得分
score_test1 = sklearn.metrics.f1_score(kpca_new_test.iloc[:,50], q1, pos_label=list(set(kpca_new_test.iloc[:,50])),average = None)
np.mean(score_test1)


results1 = []
# 最小叶子结点的参数取值
sample_leaf_options = list(range(3,11,1))
# 决策树个数参数取值
n_estimators_options=list(range(10,200,2))

#参数选择过程
for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        
        clf = ExtraTreesClassifier(max_features="log2",min_samples_leaf=leaf_size, bootstrap=True,n_estimators=n_estimators_size,n_jobs=-1,criterion="gini",oob_score=True,class_weight="balanced")
        clf.fit(kpca_new_train.iloc[:,0:50],kpca_new_train.iloc[:,50])
        q1=clf.predict(kpca_new_test.iloc[:,0:50])
#计算F1得分
        score_test = sklearn.metrics.f1_score(kpca_new_test.iloc[:,50], q1, pos_label=list(set(kpca_new_test.iloc[:,50])),average = None)
        results1.append((leaf_size,n_estimators_size,np.mean(score_test)))
#输出最佳参数组合
print(max(results1, key=lambda x:x[2]))


#使用最佳参数训练ExtraTreesClassifier模型
clf2 = ExtraTreesClassifier(max_features=None,min_samples_leaf=5,n_estimators=150, bootstrap=True,n_jobs=-1,class_weight="balanced",criterion="gini",oob_score=True)
clf2.fit(kpca_new_train.iloc[:,0:50],kpca_new_train.iloc[:,50])
q2=clf2.predict(kpca_new_test.iloc[:,0:50])
sklearn.metrics.confusion_matrix(kpca_new_test.iloc[:,50],q2)
#计算F1得分
score_test2 = sklearn.metrics.f1_score(kpca_new_test.iloc[:,50], q2, pos_label=list(set(kpca_new_test.iloc[:,50])),average = None)
np.mean(score_test2)
p2=clf2.predict(x.iloc[:0:50])




results2=[]
learning_rate_options=list(np.linspace(0.5,1,10,endpoint=False))
n_estimators_options=list(range(100,200,2))
for learning_rates in learning_rate_options:
    for n_estimatorss in n_estimators_options:
        clf3 = AdaBoostClassifier(learning_rate=learning_rates,n_estimators=n_estimatorss)
        clf3.fit(kpca_new_train.iloc[:,0:50],kpca_new_train.iloc[:,50])
        q3=clf3.predict(kpca_new_test.iloc[:,0:50])
        score_test = sklearn.metrics.f1_score(kpca_new_test.iloc[:,50], q3, pos_label=list(set(kpca_new_test.iloc[:,50])),average = None)
        results2.append((learning_rates,n_estimatorss,np.mean(score_test)))
#输出最佳参数组合
print(max(results2, key=lambda x:x[2]))
#计算F1得分
   
#训练AdaBoostClassifier模型
clf3 = AdaBoostClassifier(learning_rate=1,n_estimators=50)
clf3.fit(kpca_new_train.iloc[:,0:50],kpca_new_train.iloc[:,50])
q3=clf3.predict(kpca_new_test.iloc[:,0:50])
sklearn.metrics.confusion_matrix(kpca_new_test.iloc[:,50],q3)
#计算F1得分
score_test3 = sklearn.metrics.f1_score(kpca_new_test.iloc[:,50], q3, pos_label=list(set(kpca_new_test.iloc[:,50])),average = None)
np.mean(score_test3)
#训练GradientBoostingClassifier
clf4 = GradientBoostingClassifier(min_samples_leaf=7,learning_rate=0.8,n_estimators=150)
clf4.fit(kpca_new_train.iloc[:,0:50],kpca_new_train.iloc[:,50])
q4=clf4.predict(kpca_new_test.iloc[:,0:50])
sklearn.metrics.confusion_matrix(kpca_new_test.iloc[:,50],q4)
#计算F1得分
score_test4 = sklearn.metrics.f1_score(kpca_new_test.iloc[:,50], q4, pos_label=list(set(kpca_new_test.iloc[:,50])),average = None)
np.mean(score_test4)

