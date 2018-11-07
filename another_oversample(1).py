#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:27:35 2018

@author: heisenberg
"""

# another kind of smote oversample
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

path1 = '/Users/heisenberg/Downloads/研究生课件/天池天文竞赛/所有可使用的数据集/data.npy'
path2 = '/Users/heisenberg/Downloads/研究生课件/天池天文竞赛/所有可使用的数据集/data_another.npy'
data = np.load(path1)
data_another = np.load(path2)

w = 0.035

# N是尝试重复抽样的倍数

def NRSBoundary_smote(data, w, min_label, N, max_label=1, k=5):
    
    pData = data[(data[:,-1]==max_label) | (data[:,-1]==min_label)]
    X = pData[:,:-1]
    X_max_index = tuple(np.arange(len(X))[X[:,-1]==max_label])
    X_min_index = tuple(np.arange(len(X))[X[:,-1]==min_label])
    
    sampleSet = []
    boundSet = [] #边界域
    posSet = [] #决策正域
#    dist_k_dict = {} # 保存每个样本的k个最近临的样本的index,key也是index
    delta_dict = {} # 保存每个样本的delta值，key也是index
    
    n = len(X)
    for i in range(n):
        dist_list = []
        for j in range(n):
            if j != i:
                dist = np.sqrt(np.sum((X[i]-X[j])*(X[i]-X[j])))
                dist_list.append(dist)
        dMin = min(dist_list)
        dMax = max(dist_list)
        delta = dMin + w*dMax
        delta_dict[i] = delta
#        dist_list_sort = sorted(dist_list)[:k]
        dist_mean = sum(dist_list)/len(dist_list)
        dist_list.insert(i,dist_mean)
        delta_index_list = [i for i,value in enumerate(dist_list) if value<delta]
        delta_index_list = tuple(delta_index_list)
#        dist_k_dict[i] = [dist_list.index(k) for k in dist_list_sort]
        if (i in X_min_index) and not (delta_index_list <= X_min_index):
            boundSet.append(i)
        elif (i in X_max_index) and (delta_index_list <= X_max_index):
            posSet.append(i)
     
    
    X_min = pData[pData[:,-1]==min_label,:-1]
    neighbors = NearestNeighbors(n_neighbors=k).fit(X_min)
    
    
    for i in boundSet:
        for kk in range(N):
            isConflict = False
        #        dist_k_tuple = tuple(dist_k_dict[i])
            dist_k_list = neighbors.kneighbors(X[i].reshape(1,-1),n_neighbors=k+1,return_distance=False)[0]      
            for m in range(k):
                temp_x_index = X.index(X_min[dist_k_list[m+1]])
                x_new = X[i] + random.random()*(X[temp_x_index]-X[i])
                for j in posSet:
                    dist_new = np.sqrt(np.sum((x_new-X[j])*(x_new-X[j])))
                    if dist_new < delta_dict[j]:
                        isConflict = True
                        break          
                if isConflict == False:
                    sampleSet.append(x_new)
    return sampleSet

sampleSet2 = NRSBoundary_smote(data, 0.03, 2, N=10)
sampleSet3 = NRSBoundary_smote(data, 0.03, 3)
sampleSet4 = NRSBoundary_smote(data, 0.03, 4)

ano_sampleSet2 = NRSBoundary_smote(data_another, 0.03, 2, N=1)
ano_sampleSet3 = NRSBoundary_smote(data_another, 0.03, 3, N=1)
#ano_sampleSet4 = NRSBoundary_smote(data_another, 0.03, 4, N=1)


        
    




