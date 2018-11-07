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
from sklearn.model_selection import GridSearchCV
import pywt
import costcla
from costcla import *
#读取测试集
test_data=pd.read_csv("all_test.csv",header=None)
test_data=test_data.iloc[0:99998,:]
error=pd.read_csv("error.csv",header=None)
test_data=test_data.append(error,ignore_index=True)
test_data1=test_data.iloc[:,1:]
test_data1.columns=list(np.arange(2600))


data_train=pd.read_csv("partTrainData.csv",header=None)
mix_data=data_train.iloc[:,0:2600].append(test_data1,ignore_index=True)
mix_data=np.array(mix_data,copy=False)
#小波变换降噪(四层小波变换，然后细节系数中cd1,cd2=0,用其他系数重构光谱信号)
ca4,cd4,cd3,cd2,cd1=pywt.wavedec(mix_data,'db1',level=4)
mix_data=pywt.waverec([ca4,cd4,cd3],'db1')
mix_data=DataFrame(mix_data)
scale = StandardScaler()
x = scale.fit_transform(mix_data)#数据标准化
pca = PCA(n_components=50,whiten=True)
x=DataFrame(pca.fit_transform(x))#在各主成分投影数据,做主成分得分
test11=x.iloc[48383:,:]
train11=x.iloc[0:48383,:]
train11.loc[:,"class"]=data_train.iloc[:,2600]
train11_train,train11_test=train_test_split(train11,test_size=0.3)#分割训练集测试集
#参数寻优
results1 = []
# 最小叶子结点的参数取值
sample_leaf_options = list(range(1,11,1))
# 决策树个数参数取值
n_estimators_options=list(range(190,400,3))
max_features_options=list(range(1,51,2))
max_depth_options=list(range(10,50,2))
min_samples_split_options=list(range(2,100,2))

#参数选择过程
for leaf_size in sample_leaf_options:
    for n_estimators_size in n_estimators_options:
        for max_features in max_features_options:
            for max_depth_size in max_depth_options:
                    
                for min_samples_splits in min_samples_split_options:
                    for n_estimators_size in n_estimators_options:
                        
                
            
        
                        clf = RandomForestClassifier(max_features=19,min_samples_leaf=8,n_estimators=n_estimators_size,n_jobs=-1,criterion="gini",oob_score=True,class_weight="balanced")
                        clf.fit(train11_train.iloc[:,0:50],train11_train.iloc[:,50])
                        q1=clf.predict(train11_test.iloc[:,0:50])
#计算F1得分
                        score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50], q1, pos_label=list(set(train11_test.iloc[:,50])),average = None)
                        results1.append((n_estimators_size ,np.mean(score_test)))
#输出最佳参数组合
print(max(results1, key=lambda x:x[1]))


#使用最优参数建模
clf1 =  RandomForestClassifier(min_samples_leaf=4,max_features=29,
                             n_estimators=300,max_depth=50,n_jobs=-1,
                             criterion="gini",oob_score=True,class_weight="balanced",random_state=4,bootstrap=True)
clf1.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
q1=clf1.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q1)
score_test1 = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q1, pos_label=list(set(train11_test.iloc[:,50].values)),average =None)
np.mean(score_test1)

#参数寻优
parameters2 = [ {'n_estimators': list(range(100,300,2)),

"max_features":list(range(7,50,2)),

'max_depth': list(range(1,10,1)) ,

'min_samples_split':list(range(1,10,1)) ,

'min_weight_fraction_leaf': [ 0.0,0.1,0.2,0.3,0.4,0.5 ]}]

clf2 = ExtraTreesClassifier(max_features=29,max_depth=50,bootstrap=True,n_jobs=-1,class_weight="balanced",oob_score=True)
grid_search2 = GridSearchCV(clf2, parameters2, n_jobs=-1,verbose=1, scoring='f1_macro',cv=5)
grid_search2.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)

#使用最佳参数训练ExtraTreesClassifier模型

clf2 = ExtraTreesClassifier(max_features=29,max_depth=50,min_samples_leaf=4,min_samples_split=12,random_state =4,n_estimators=265, bootstrap=True,n_jobs=-1,class_weight="balanced",criterion="gini",oob_score=True)
clf2.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
q2=clf2.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q2)
score_test2 = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q2, pos_label=list(set(train11_test.iloc[:,50].values)),average = None)
np.mean(score_test2)

#参数寻优
results2=[]
learning_rate_options=list(np.linspace(0.5,0.6,20,endpoint=True))
n_estimators_options=list(range(100,200,2))
min_samples_split_options=list(range(2,50,2))


    for n_estimatorss in n_estimators_options:
        for min_samples_splits in min_samples_split_options:
            for learning_rates in learning_rate_options:
                clf3 = GradientBoostingClassifier(min_samples_split=12,min_samples_leaf=10,learning_rate=learning_rates,n_estimators=144,max_features=19)
                clf3.fit(train11_train.iloc[:,0:50],train11_train.iloc[:,50])
                q3=clf3.predict(train11_test.iloc[:,0:50])
                score_test = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q3, pos_label=list(set(train11_test.iloc[:,50].values)),average = None)
                results2.append((learning_rates,np.mean(score_test)))
#输出最佳参数组合
print(max(results2, key=lambda x:x[1]))

#训练GradientBoostingClassifier

clf4 = GradientBoostingClassifier(random_state=6,max_features=19,min_samples_split=10,min_samples_leaf=10,learning_rate=0.55263,n_estimators=144)
clf4.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
q4=clf4.predict(train11_test.iloc[:,0:50].values)
sklearn.metrics.confusion_matrix(train11_test.iloc[:,50].values,q4)
#计算F1得分
score_test4 = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q4, pos_label=list(set(train11_test.iloc[:,50].values)),average = None)
np.mean(score_test4)


#模型融合


    RF =  RandomForestClassifier(max_features =None,n_estimators=52, n_jobs=-1,class_weight="balanced",bootstrap=True,criterion="gini",oob_score=True)
    sclf = StackingCVClassifier(classifiers=[clf1,clf2,clf4], 
                          meta_classifier=RF,cv=5,use_probas=True)#0.57271159041579445  52
    sclf.fit(train11_train.iloc[:,0:50].values,train11_train.iloc[:,50].values)
    q5=sclf.predict(train11_test.iloc[:,0:50].values)
    score_test5 = sklearn.metrics.f1_score(train11_test.iloc[:,50].values, q5, pos_label=list(set(train11_test.iloc[:,50])),average = None)
    np.mean(score_test5)








p2=clf1.predict(test11.iloc[:,0:50])
Counter(p2)
test11.loc[:,"class"]=p2

test11.reset_index(drop=True,inplace=True)
test11.loc[:,"label"]=test_data.iloc[:,0]

test11.loc[:,["label","class"]].to_csv("test11c.csv")