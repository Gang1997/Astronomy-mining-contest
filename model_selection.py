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
from sklearn.svm import OneClassSVM
import copy
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
pca = PCA(n_components=50,whiten=True)
x_pca_new=pca.fit_transform(x)#在各主成分投影数据，做主成分得分
x_pca_new=DataFrame(x_pca_new)
test11=x_pca_new.iloc[48383:,:]
train11=x_pca_new.iloc[0:48383,:]
train11.loc[:,"class"]=data_train.iloc[:,2600]
train11_train,train11_test=train_test_split(train11,test_size=0.3,random_state=1)#分割训练集测试集
#针对star训练
class1_train=copy.copy(train11_train)
class1_test=copy.copy(train11_test)
a1=class1_train.loc[:,'class']
b1=class1_test.loc[:,'class']
a1[a1!=1]=-1
b1[b1!=1]=-1
nu1=len(a1[a1==-1])/len(a1)
clf1=OneClassSVM('rbf',nu=nu1,random_state=30,gamma=0.001,shrinking=True)
clf1.fit(class1_train[a1==1].iloc[:,0:50].values)
class1=clf1.predict(class1_test[b1==1].iloc[:,0:50].values)
#针对galaxy训练
class2_train=copy.copy(train11_train)
class2_test=copy.copy(train11_test)
a2=class2_train.loc[:,'class']
b2=class2_test.loc[:,'class']
a2[a2!=2]=1
a2[a2==2]=-1
b2[b2!=2]=1
b2[b2==2]=-1
nu2=len(a2[a2==-1])/len(a2)
clf2=OneClassSVM('rbf',nu=nu2,random_state=30,gamma=0.001,shrinking=True)
clf2.fit(class2_train[a2==1].iloc[:,0:50].values)
class2=clf2.predict(class2_test[b2==1].iloc[:,0:50].values)
#针对galaxy的训练
class3_train=copy.copy(train11_train)
class3_test=copy.copy(train11_test)
a3=class3_train.loc[:,'class']
b3=class3_test.loc[:,'class']
a3[a3!=3]=1
a3[a3==3]=-1
b3[b3!=3]=1
b3[b3==3]=-1
nu3=len(a3[a3==-1])/len(a3)
clf3=OneClassSVM('rbf',nu=nu3,random_state=30,gamma=0.001,shrinking=True)
clf3.fit(class3_train[a3==1].iloc[:,0:50].values)
class3=clf3.predict(class3_test[b3==1].iloc[:,0:50].values)
#针对unknown的训练
class4_train=copy.copy(train11_train)
class4_test=copy.copy(train11_test)
a4=class4_train.loc[:,'class']
b4=class4_test.loc[:,'class']
a4[a4!=4]=1
a4[a4==4]=-1
b4[b4!=4]=1
b4[b4==4]=-1
nu4=len(a4[a4==-1])/len(a4)
clf4=OneClassSVM('rbf',nu=nu4,random_state=30,gamma=0.001,shrinking=True)
clf4.fit(class4_train[a4==1].iloc[:,0:50].values)
class4=clf4.predict(class4_test[b4==1].iloc[:,0:50].values)
def classier(test_data):
    pre1=clf1.predict(test_data.iloc[:,0:50])
    test_data['pre1']=pre1
    test_data.ix[test_data['pre1']==1,'pre1']='star'   
    pre2=clf2.predict(test_data.ix[test_data['pre1']==-1,0:50])
    test_data.ix[test_data['pre1']==-1,'pre1']=pre2
    test_data.ix[test_data['pre1']==-1,'pre1']='galaxy'
    pre3=clf3.predict(test_data.ix[test_data['pre1']==1,0:50])
    test_data.ix[test_data['pre1']==1,'pre1']=pre3
    test_data.ix[test_data['pre1']==-1,'pre1']='qso'
    test_data.ix[test_data['pre1']==1,'pre1']='unknown'
    return test_data
test=classier(test11)

           
  

        
        
    


 
  



