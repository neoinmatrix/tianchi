# coding=utf-8
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import time
from dateutil.parser import parse

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
import lightgbm as lgb

# from sklearn.metrics import mean_absolute_error as mae
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
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
    for train_index, test_index in kf.split(train.index):
        d_tx=train.loc[train_index,x_cols].values
        d_ty=train.loc[train_index,y_cols].values.ravel()
        d_cx=train.loc[test_index,x_cols].values
        d_cy=train.loc[test_index,y_cols].values.ravel()

        lgb_train = lgb.Dataset(d_tx, d_ty)
        lgb_eval = lgb.Dataset(d_cx, d_cy, reference=lgb_train)

        lgb_params = {
            'learning_rate': 0.01,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mse',
            # 'sub_feature': 0.7,
            'num_leaves': 60,
            # 'colsample_bytree': 0.7,
            'feature_fraction': 0.7,
            'min_data': 100,
            'min_hessian': 1,
            'verbose': -1,
            "verbosity" :-1,
        }

        gbm = lgb.train(lgb_params,lgb_train,
                        num_boost_round=20,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=5)
        p_cy= gbm.predict(d_cx)        #, num_iteration=gbm.best_iteration
        p_test[:,kf_times]= gbm.predict(test.loc[:,x_cols].values) #, num_iteration=gbm.best_iteration
        
        mse_arr[kf_times]=mse(p_cy,d_cy)*0.5
        if params["debug"]==True:
            print "kf-%d-mse:%.4f"%(kf_times,mse_arr[kf_times])
        kf_times=kf_times+1

    print "the train time:%f"%(time.time()-startTime)
    print "the train hmse:",mse_arr.mean(),mse_arr.std()
    if params["saved"]==True:
        fn="./data/%s_all.csv"%time.strftime("%Y%m%d_%H%M%S", time.localtime())
        np.savetxt(fn,p_test,fmt='%.4f')
        sub = pd.DataFrame({'pred':p_test.mean(axis=1)})
        fn="./data/%s.csv"%time.strftime("%Y%m%d_%H%M%S", time.localtime())
        sub.to_csv(fn, header=None, index=False, float_format='%.4f')

if __name__=="__main__":
    train,test=getData()
    remove=[u'血糖',u'id',
    # u'乙肝e抗体',u'乙肝e抗原',u'乙肝核心抗体',u'乙肝表面抗体',u'乙肝表面抗原'
    ]
    # x_cols=[f for f in train.columns if f not in remove]

    x_cols=[
        u'年龄',
        u'*天门冬氨酸氨基转换酶',
        u'尿酸',
        u'*r-谷氨酰基转换酶',
        u'*丙氨酸氨基转换酶',
        u'*碱性磷酸酶',
    ]
    y_cols=u'血糖'
    params={
        'kf_n':5,'train':train,'test':test,
        'x_cols':x_cols,'y_cols':y_cols,
        'debug':True,'saved':True,
    }
    frm(params)
    print "ok"