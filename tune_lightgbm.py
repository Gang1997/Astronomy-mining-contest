#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 09:50:56 2018

@author: heisenberg
"""

# tune xgboost
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
import time


# 获取测试剂
test_data=pd.read_csv("/Users/heisenberg/Downloads/研究生课件/天池天文竞赛/all.csv",header=None)
test_data=test_data.iloc[0:99998,:]

error=pd.read_csv("/Users/heisenberg/Downloads/研究生课件/天池天文竞赛/error.csv",header=None)
test_data=test_data.append(error,ignore_index=True)
test_data1=test_data.iloc[:,1:]
test_data1.columns=list(np.arange(2600))
test_label = test_data.iloc[:,0]
np.save('/Users/heisenberg/Downloads/研究生课件/天池天文竞赛/所有可使用的数据集/test_data.npy',np.array(test_data1))
np.save('/Users/heisenberg/Downloads/研究生课件/天池天文竞赛/所有可使用的数据集/test_num.npy',np.array(test_label))

# 采用完整少数标签，先过采样后pca的数据
path0 = '/Users/heisenberg/Downloads/研究生课件/天池天文竞赛/Data20.npy'
path1 = '/Users/heisenberg/Downloads/研究生课件/天池天文竞赛/所有可使用的数据集/test_data.npy'
data_train = np.load(path0)
data_test = np.load(path1)
label_column = [np.NaN]*len(data_test)
data_test = np.c_[data_test, label_column]

data = np.concatenate((data_train, data_test))

X, y = data[:,:-1], data[:,-1]
#X_train, y_train = data_train[:,:-1], data_train[:,-1]
#X_test, y_test = data_test[:,:-1], data_test[:,-1]

scale = StandardScaler()
X = scale.fit_transform(X)

pca = PCA(n_components=50)
pca.fit(X)
X_train = pca.transform(X[:96770])
X_test = pca.transform(X[96770:])
y = y.astype(int)
y[y==4] = 0
y_train = y[:96770]
y_test = y[96770:]

X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
lgb_train = lgb.Dataset(X_train_, y_train_)
lgb_eval = lgb.Dataset(X_test_, y_test_, reference=lgb_train)


# 两种调参方式，一种是写循环，利用lgb.cv调参；一种是利用GridSearchCV，基于lgb.LGBMClassifier调参。
# 关键是两种方式的同一种参数的名称却不同


# 循环 + lgb.cv

params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 4,
        'metric': {'multi_error'},
        'max_depth': -1,
        'learning_rate': 0.3, # 1,  0.01-0.2
        'num_leaves': 160, # 2，树深度参数，待调 
        'min_child_samples': 130, # 2，一个叶子上最小的数据量，处理过拟合，待调 
        'min_sum_hessian_in_leaf': 1e-3, # 类似与上个参数，调整上一个就不调整这个了
        'min_split_gain': 1e-5,
        'feature_fraction': 0.9, # 3，每棵树训练选取的特征，类似于max_features
        'bagging_fraction': 0.8, # 3，不进行重采样的情况下随机选择部分数据，类似于subsample，小的话防止overfitting
        'bagging_freq': 4,
        'nthreads': -1,
        'verbose': 1,
    }
params['is_unbalance']='true'

best_params = {}


min_merror = float('Inf')

start = time.time()
for num_boost_round in range(70,121,10):
    cv_results = lgb.cv(params,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        seed=42,
                        nfold=5,
                        metrics=['multi_error'],
                        early_stopping_rounds=15
                          )
    mean_merror = pd.Series(cv_results['multi_error-mean']).min()
    boost_rounds = pd.Series(cv_results['multi_error-mean']).argmin()
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params['num_boost_round'] = boost_rounds
end = time.time()
time_ = end - start
print(time_)

# 暂定learning_rate =0.1, num_boost_round=78
# 现调整num_leaves和min_child_samples
min_merror = float('Inf')
start = time.time()
for num_leaves in [15, 31, 63, 127, 160, 255]:
    for min_child_samples in range(10, 191, 20):
        params['num_leaves'] = num_leaves
        params['min_child_samples'] = min_child_samples
        
        cv_results = lgb.cv(params, lgb_train, 
                            num_boost_round=best_params['num_boost_round'],
                            seed=42, nfold=5, metrics=['multi_error'],
                            early_stopping_rounds=10)
        
        mean_merror = pd.Series(cv_results['multi_error-mean']).min()
        
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['min_child_samples'] = min_child_samples
end = time.time()
time_ = end - start
print(time_)

params['num_leaves'] = best_params['num_leaves']
params['min_child_samples'] = best_params['min_child_samples']

# 现调整feature_fraction和bagging_fraction

min_merror = float('Inf')
start = time.time()
for feature_fraction in [0.75, 0.8, 0.85, 0.9, 0.95]:
    for bagging_fraction in [0.75,0.8,0.85,0.9,0.95]:
        params['feature_fraction'] = feature_fraction
        params['bagging_fraction'] = bagging_fraction
        
        cv_results = lgb.cv(params, lgb_train, 
                            num_boost_round=best_params['num_boost_round'],
                            seed=42, nfold=5, metrics=['multi_error'],
                            early_stopping_rounds=10)
        
        mean_merror = pd.Series(cv_results['multi_error-mean']).min()
        
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['bagging_fraction'] = bagging_fraction
            best_params['feature_fraction'] = feature_fraction
end = time.time()
time_ = end - start
print(time_)

params['bagging_fraction'] = best_params['bagging_fraction']
params['feature_fraction'] = best_params['feature_fraction']


# 现调整min_split_gain
min_merror = float('Inf')
start = time.time()
for min_split_gain in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    params['min_split_gain'] = min_split_gain
    
    cv_results = lgb.cv(params, lgb_train, 
                            num_boost_round=best_params['num_boost_round'],
                            seed=42, nfold=5, metrics=['multi_error'],
                            early_stopping_rounds=10)
    
    mean_merror = pd.Series(cv_results['multi_error-mean']).min()
        
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params['min_split_gain'] = min_split_gain

end = time.time()
time_ = end - start
print(time_)

min_merror = float('Inf')
start = time.time()
for min_split_gain in [1e-10, 1e-8, 1e-12]:
    params['min_split_gain'] = min_split_gain
    
    cv_results = lgb.cv(params, lgb_train, 
                            num_boost_round=best_params['num_boost_round'],
                            seed=42, nfold=5, metrics=['multi_error'],
                            early_stopping_rounds=10)
    
    mean_merror = pd.Series(cv_results['multi_error-mean']).min()
        
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params['min_split_gain'] = min_split_gain

end = time.time()
time_ = end - start
print(time_)

params['min_split_gain'] = best_params['min_split_gain']

#################################
# 降低学习率继续调整
params['learning_rate'] = 0.05
min_merror = float('Inf')
best_params = {}

start = time.time()
for num_boost_round in range(40,201,10):
    cv_results = lgb.cv(params,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        seed=42,
                        nfold=5,
                        metrics=['multi_error'],
                        early_stopping_rounds=15
                          )
    mean_merror = pd.Series(cv_results['multi_error-mean']).min()
    boost_rounds = pd.Series(cv_results['multi_error-mean']).argmin()
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params['num_boost_round'] = boost_rounds
end = time.time()
time_ = end - start
print(time_)

min_merror = float('Inf')
start = time.time()
for num_leaves in [31,41,51,63,73,82]:
    for min_child_samples in range(30, 70, 5):
        params['num_leaves'] = num_leaves
        params['min_child_samples'] = min_child_samples
        
        cv_results = lgb.cv(params, lgb_train, 
                            num_boost_round=best_params['num_boost_round'],
                            seed=42, nfold=5, metrics=['multi_error'],
                            early_stopping_rounds=10)
        
        mean_merror = pd.Series(cv_results['multi_error-mean']).min()
        
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['min_child_samples'] = min_child_samples
end = time.time()
time_ = end - start
print(time_)

params['min_child_samples'] = 45

for num_leaves in [82,100,127]:
    params['num_leaves'] = num_leaves
    cv_results = lgb.cv(params, lgb_train, 
                                num_boost_round=best_params['num_boost_round'],
                                seed=42, nfold=5, metrics=['multi_error'],
                                early_stopping_rounds=10)
        
    mean_merror = pd.Series(cv_results['multi_error-mean']).min()
    
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params['num_leaves'] = num_leaves
        
params['num_leaves'] = best_params['num_leaves']
params['min_child_samples'] = 80

min_merror = float('Inf')
start = time.time()
for feature_fraction in [0.6, 0.7, 0.8, 0.9]:
    for bagging_fraction in [0.6,0.7,0.8,0.9]:
        params['feature_fraction'] = feature_fraction
        params['bagging_fraction'] = bagging_fraction
        
        cv_results = lgb.cv(params, lgb_train, 
                            num_boost_round=best_params['num_boost_round'],
                            seed=42, nfold=5, metrics=['multi_error'],
                            early_stopping_rounds=10)
        
        mean_merror = pd.Series(cv_results['multi_error-mean']).min()
        
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['bagging_fraction'] = bagging_fraction
            best_params['feature_fraction'] = feature_fraction
end = time.time()
time_ = end - start
print(time_)

params['bagging_fraction'] = best_params['bagging_fraction']
params['feature_fraction'] = best_params['feature_fraction']


# 在test数据集上测试macro-f1值
from sklearn.metrics import  precision_recall_fscore_support
from collections import Counter


def calMacroF1(params, best_params = best_params, lgb_train=lgb_train, lgb_eval=lgb_eval, X_test=X_test_, y_test=y_test_):
    
    model = lgb.train(params, lgb_train, num_boost_round=best_params['num_boost_round']+300,
                      valid_sets=lgb_eval, early_stopping_rounds=30)
    
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_label = np.argmax(y_pred, axis=1)
    
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred_label, labels=[0, 1, 2, 3], average='macro')
        
    print(f_macro)


calMacroF1(params)


model = lgb.train(params, lgb_train, num_boost_round=best_params['num_boost_round'],
                      valid_sets=lgb_eval, early_stopping_rounds=30)


y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_label = np.argmax(y_pred, axis=1)
Counter(y_pred_label)


















