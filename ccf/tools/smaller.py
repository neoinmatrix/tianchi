# coding=utf-8
import numpy as np 
import pandas as pd

def smallfile(filex,files,size=100):
    predicts = pd.read_csv(filex) 
    predicts=predicts.loc[0:size,:]
    predicts.to_csv(files)

file_arr=[
    "ccf_offline_stage1_test_revised.csv",
    "ccf_offline_stage1_train.csv",
    "ccf_online_stage1_train.csv",
]
file_small=[
    "predicts.csv",
    "offline_train.csv",
    "online_train.csv",
]
for r,s in zip(file_arr,file_small):
    smallfile('../'+r,'../'+s)
print "ok"