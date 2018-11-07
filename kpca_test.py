# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:55:20 2018

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

test_data1=pd.read_csv("data1_test.csv",header=None)
p2=clf1.predict(test_data1.iloc[:,1:])
test_data1.loc[:,"class"]=p2
Counter(p2)
test_data1.to_csv("test1.csv")

