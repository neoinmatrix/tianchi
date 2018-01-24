# coding=utf-8
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import time
from dateutil.parser import parse

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.neural_network import MLPRegressor

# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error as mae
# from sklearn.svm import SVR
# from sklearn.decomposition import PCA

def getData():
    test="../data/d_test_A_20180102.csv"
    train="../data/d_train_20180102.csv"
    train=pd.read_csv(train, encoding="gbk")
    test=pd.read_csv(test, encoding="gbk")
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    df = pd.concat([train,test])
    df[u'性别'] = df[u'性别'].map({u'男':1,u'女':0})
    df[u'体检日期'] = pd.to_datetime(df[u'体检日期'])
    df[u'体检日期'] = (df[u'体检日期']-parse('2017-09-10')).dt.days
    df.fillna(df.median(axis=0),inplace=True)
    train_feat = df[df.id.isin(train_id)]
    test_feat = df[df.id.isin(test_id)]
    return train_feat,test_feat

if __name__=="__main__":
    train,test=getData()
    print train.head()
    print "ok"