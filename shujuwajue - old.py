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
import scipy.signal as signal
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
data_train=pd.read_csv("partTrainData.csv",header=None)
data_train.rename(columns={2600:"class"},inplace=True)
num=[]
for i in range(len(data_train)):
    c=list(data_train.iloc[i])
    a=c.count(0)
    num.append(a)
data_train["count0"]=num#计算每行0的个数
data_train1=data_train.loc[data_train["count0"]>5]
data_train1.reset_index(inplace=True)
data_train1.sort_values("count0")
x_data_train1=data_train1.iloc[:,1:2601]
y_data_train1=data_train1.iloc[:,2601]
data_train2=data_train.loc[data_train["count0"]<=5]
data_train2.reset_index(inplace=True)
x_data_train2=data_train2.iloc[:,1:2601]
y_data_train2=data_train2.iloc[:,2601]
#首先对0较少的数据集建模
#Lxp_train=x_data_train2.apply(lambda x: signal.medfilt(x, kernel_size=29), axis=1)
#x_data_train2_a=x_data_train2/Lxp_train
#x_data_train2_b=x_data_train2_a.apply(lambda x: signal.medfilt(x, kernel_size=9), axis=1)
from sklearn.decomposition import KernelPCA
np.random.seed(0)
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x = scale.fit_transform(x_data_train2)#数据标准化
x_data=DataFrame(x)
kpca = KernelPCA(kernel='rbf',copy_X=False)
x_kpca_new=kpca.fit_transform(x_data)#在各核主成分投影数据，做主成分得分
kpca_new=pd.concat([x_kpca_new,y_data_train2],axis=1)
kpca_new_test,kpca_new_train=train_test_split(kpca_new,test_size=0.3,random_state = 0)#分割训练集测试集
#pca技术
from sklearn.decomposition import PCA
np.random.seed(0)
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x = scale.fit_transform(x_data_train2)#数据标准化
x_data=DataFrame(x)
pca = PCA(n_components=50)
x_pca_new=pca.fit_transform(x_data)#在各主成分投影数据，做主成分得分
x_pca=DataFrame(x_pca_new)
pca_new=pd.concat([x_pca,y_data_train2],axis=1)
pca_new_test,pca_new_train=train_test_split(pca_new,test_size=0.3)#分割训练集测试集
#随机森林预测模型
data_train2_test,data_train2_train=train_test_split(data_train2,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
results = []
# 最小叶子结点的参数取值
sample_leaf_options = list(range(5,15,1))
# 决策树个数参数取值
n_estimators_options=list(range(190,210,1))


for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        clf = RandomForestClassifier(max_features=None,min_samples_leaf=leaf_size, n_estimators=n_estimators_size,criterion="gini",class_weight="balanced")
        clf.fit(pca_new_train.iloc[:,0:50],pca_new_train.iloc[:,50])
        q1=clf.predict(pca_new_test.iloc[:,0:50])
#计算F1得分
        score_test = sklearn.metrics.f1_score(pca_new_test.iloc[:,50], q1, pos_label=list(set(pca_new_test.iloc[:,50])),average = None)
        results.append((leaf_size,n_estimators_size,np.mean(score_test)))
print(max(results, key=lambda x:x[2]))

x=np.load("t.npy")
x_kpca=DataFrame(x)
pca_new_test,pca_new_train=train_test_split(pca_new,test_size=0.3)#分割训练集测试集

clf1 = RandomForestClassifier(max_features="log2",min_samples_leaf=6,n_estimators=300, n_jobs=-1, criterion="gini",class_weight="balanced")
clf1.fit(pca_new_train.iloc[:,0:50],pca_new_train.iloc[:,50])
q1=clf1.predict(pca_new_test.iloc[:,0:50])
sklearn.metrics.confusion_matrix(pca_new_test.iloc[:,50],q1)
#计算F1得分
score_test = sklearn.metrics.f1_score(pca_new_test.iloc[:,50], q1, pos_label=list(set(pca_new_test.iloc[:,50])),average = None)
np.mean(score_test)
