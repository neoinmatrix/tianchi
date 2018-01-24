# coding=utf-8
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import time
from dateutil.parser import parse

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error as mae
# from sklearn.svm import SVR
# from sklearn.decomposition import PCA

def getData():
    test="./data/d_test_A_20180102.csv"
    train="./data/d_train_20180102.csv"
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

def frm(params):
    startTime = time.clock()
    train=params["train"]
    test=params["test"]
    x_cols=params["x_cols"]
    y_cols=params["y_cols"]

    mse_arr = np.zeros(params['kf_n'])
    p_test = np.zeros((test.shape[0],params['kf_n']))
    kf_times=0
    kf = KFold(n_splits=params['kf_n'], shuffle=True,random_state=np.random.randint(11))
    
    alpha=params["alpha"]
    layer=params["layer"]
    for train_index, test_index in kf.split(train.index):
        d_tx=train.loc[train_index,x_cols].values
        d_ty=train.loc[train_index,y_cols].values.ravel()
        d_cx=train.loc[test_index,x_cols].values
        d_cy=train.loc[test_index,y_cols].values.ravel()

        clf = MLPRegressor(alpha=alpha, hidden_layer_sizes=layer, random_state=1)
        clf.fit(d_tx, d_ty) 
        p_cy = clf.predict(d_cx)
        p_test[:,kf_times]= clf.predict(test.loc[:,x_cols].values) #, num_iteration=gbm.best_iteration
        
        mse_arr[kf_times]=mse(p_cy,d_cy)*0.5
        if params["debug"]==True:
            print "kf-%d-mse:%.4f"%(kf_times,mse_arr[kf_times])
        kf_times=kf_times+1

    print "the train time:%f"%float(time.clock() - startTime)
    print "the train hmse:",mse_arr.mean(),mse_arr.std()
    if params["saved"]==True:
        fn="./data/%s_all.csv"%time.strftime("%Y%m%d_%H%M", time.localtime())
        np.savetxt(fn,p_test,fmt='%.4f')
        sub = pd.DataFrame({'pred':p_test.mean(axis=1)})
        fn="./data/%s.csv"%time.strftime("%Y%m%d_%H%M", time.localtime())
        sub.to_csv(fn, header=None, index=False, float_format='%.4f')

def frm_clf(params):
    startTime = time.clock()
    train=params["train"]
    test=params["test"]
    x_cols=params["x_cols"]
    y_cols=params["y_cols"]

    mse_arr = np.zeros(params['kf_n'])
    p_test = np.zeros((test.shape[0],params['kf_n']))
    kf_times=0
    kf = KFold(n_splits=params['kf_n'], shuffle=True,random_state=np.random.randint(11))
    
    alpha=params["alpha"]
    layer=params["layer"]
    for train_index, test_index in kf.split(train.index):
        d_tx=train.loc[train_index,x_cols].values
        d_ty=train.loc[train_index,y_cols].values.ravel()
        d_cx=train.loc[test_index,x_cols].values
        d_cy=train.loc[test_index,y_cols].values.ravel()

        clf = MLPRegressor(alpha=alpha, hidden_layer_sizes=layer, random_state=1)
        clf.fit(d_tx, d_ty) 
        p_cy = clf.predict(d_cx)
        p_test[:,kf_times]= clf.predict(test.loc[:,x_cols].values) #, num_iteration=gbm.best_iteration
        
        mse_arr[kf_times]=mse(p_cy,d_cy)*0.5
        if params["debug"]==True:
            print "kf-%d-mse:%.4f"%(kf_times,mse_arr[kf_times])
        kf_times=kf_times+1

    print "the train time:%f"%float(time.clock() - startTime)
    print "the train hmse:",mse_arr.mean(),mse_arr.std()
    if params["saved"]==True:
        fn="./data/%s_all.csv"%time.strftime("%Y%m%d_%H%M", time.localtime())
        np.savetxt(fn,p_test,fmt='%.4f')
        sub = pd.DataFrame({'pred':p_test.mean(axis=1)})
        fn="./data/%s.csv"%time.strftime("%Y%m%d_%H%M", time.localtime())
        sub.to_csv(fn, header=None, index=False, float_format='%.4f')


x_cols=[
   # u'年龄', 
   u'*天门冬氨酸氨基转换酶', u'*丙氨酸氨基转换酶', u'*碱性磷酸酶',
   u'*r-谷氨酰基转换酶',
   u'*总蛋白', u'白蛋白', u'*球蛋白', u'白球比例', 
   u'甘油三酯', u'总胆固醇', u'高密度脂蛋白胆固醇', 
   u'低密度脂蛋白胆固醇', u'尿素', u'肌酐', u'尿酸', 
   u'红细胞计数', u'血红蛋白', u'红细胞压积',
   u'红细胞平均体积', u'红细胞平均血红蛋白量', u'红细胞平均血红蛋白浓度', 
   # u'红细胞体积分布宽度', u'血小板计数', u'血小板平均体积', u'血小板体积分布宽度', u'血小板比积', 
   # u'中性粒细胞%', u'淋巴细胞%', u'单核细胞%',
   # u'嗜酸细胞%', u'嗜碱细胞%'
]
if __name__=="__main__":
    train,test=getData()

    ntrain=train[train[u"血糖"]<7.0]
    ntrain=ntrain.reset_index()
    ntrain=ntrain.reset_index()
    untrain=train[train[u"血糖"]>=7.0]
    untrain=untrain.reset_index()


    # train.loc[train[u"血糖"]<7.0,u"血糖"]=0
    # train.loc[train[u"血糖"]>=7.0,u"血糖"]=1
    train["ill"]=train[u"血糖"]<7.0
    train["ill"] = train["ill"].map({True:0,False:1})
    a=(train[x_cols]-train[x_cols].min())/(train[x_cols].max()-train[x_cols].min())
    b=(test[x_cols]-test[x_cols].min())/(test[x_cols].max()-test[x_cols].min())

    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(30,30), random_state=1)
    clf.fit(a.values, train["ill"].values.ravel()) 
    p_cy = clf.predict(b.values)


    res=np.zeros([1000])
    a=(ntrain[x_cols]-ntrain[x_cols].min())/(ntrain[x_cols].max()-ntrain[x_cols].min())
    b=(untrain[x_cols]-untrain[x_cols].min())/(untrain[x_cols].max()-untrain[x_cols].min())

    reg1 = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(30,30), random_state=1)
    reg2 = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(30,30), random_state=1)
    reg1.fit(a.values, ntrain[u"血糖"].values.ravel()) 
    reg2.fit(b.values, untrain[u"血糖"].values.ravel()) 
    
    tmp=test.loc[p_cy<1,x_cols]
    a=(tmp[x_cols]-tmp[x_cols].min())/(tmp[x_cols].max()-tmp[x_cols].min())
    res[p_cy<1]= reg1.predict(a.values)


    tmp=test.loc[p_cy>0,x_cols]
    a=(tmp[x_cols]-tmp[x_cols].min())/(tmp[x_cols].max()-tmp[x_cols].min())
    res[p_cy>0]= reg2.predict(a.values)


    sub = pd.DataFrame({'pred':res})
    fn="./data/%s.csv"%time.strftime("%Y%m%d_%H%M", time.localtime())
    sub.to_csv(fn, header=None, index=False, float_format='%.4f')
        
    print "ok"