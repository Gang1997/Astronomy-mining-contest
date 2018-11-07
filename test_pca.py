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

#首先训练pca模型
train_data=np.load("data_oversample.npy")
train_data=DataFrame(train_data)
scale = StandardScaler()
x = scale.fit_transform(train_data.iloc[:,0:2600])#数据标准化
x=DataFrame(x)
pca = PCA(n_components=50)
x_pca_new=pca.fit_transform(x)#在各主成分投影数据，做主成分得分
x_pca_new=DataFrame(x_pca_new)
x_pca_new.loc[:,"class"]=train_data.iloc[:,2600]
#转化test主成分得分
scale = StandardScaler()
x_test= scale.fit_transform(test_data.iloc[:,1:])#数据标准化
test_pca=pca.transform(x_test)
test_pca=DataFrame(test_pca)
test_pca.loc[:,"label"]=test_data.iloc[:,0]


#训练模型
x_pca_new_test,x_pca_new_train=train_test_split(x_pca_new,test_size=0.3)#分割训练集测试集

results1 = []
# 最小叶子结点的参数取值
sample_leaf_options = list(range(3,11,1))
# 决策树个数参数取值
n_estimators_options=list(range(150,190,1))

#参数选择过程
for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        
        clf = ExtraTreesClassifier(max_features="log2",min_samples_leaf=leaf_size, bootstrap=True,n_estimators=n_estimators_size,n_jobs=-1,criterion="gini",oob_score=True,class_weight="balanced")
        clf.fit(x_pca_new_train.iloc[:,0:50],x_pca_new_train.iloc[:,50])
        q1=clf.predict(x_pca_new_test.iloc[:,0:50])
#计算F1得分
        score_test1 = sklearn.metrics.f1_score(x_pca_new_test.iloc[:,50], q1, pos_label=list(set(x_pca_new_test.iloc[:,50])),average = None)
        results1.append((leaf_size,n_estimators_size,np.mean(score_test1)))
#输出最佳参数组合
print(max(results1, key=lambda x:x[2]))
#使用最佳参数训练ExtraTreesClassifier模型
clf1 = RandomForestClassifier(max_features=None,min_samples_leaf=3,n_estimators=173, n_jobs=-1,class_weight="balanced",criterion="gini",oob_score=True)
clf1.fit(x_pca_new_train.iloc[:,0:50],x_pca_new_train.iloc[:,50])
q1=clf1.predict(x_pca_new_test.iloc[:,0:50])
sklearn.metrics.confusion_matrix(x_pca_new_test.iloc[:,50],q1)
#计算F1得分
score_test1 = sklearn.metrics.f1_score(x_pca_new_test.iloc[:,50], q1, pos_label=list(set(x_pca_new_test.iloc[:,50])),average = None)
np.mean(score_test1)

p2=clf1.predict(test_pca.iloc[:,0:50])
test_pca.loc[:,"class"]=p2
Counter(p2)
test_pca.loc[:,["label","class"]].to_csv("test.csv")

