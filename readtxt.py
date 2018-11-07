# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:36:55 2018

@author: YUBO
import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import time
filename=os.listdir()
data_test=pd.DataFrame(np.arange(2601).reshape(1,2601))
start=time.time()
for file_name in filename:
    data=pd.read_table(file_name,sep=',',header=None)
    file_name1=pd.DataFrame({0:[file_name})
    data2=pd.concat([file_name1,data],axis=1,ignore_index=True)
    data_test=data_test.append(data2,ignore_index=True)
    

end=time.time()
ctime=end-start  
    

data_test.to_csv("data_test11.csv")

a=pd.DataFrame()
