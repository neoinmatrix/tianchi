# coding=utf-8
import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


data_path = './data/'

train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb2312')
test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb2312')

def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])

    data[u'性别'] = data[u'性别'].map({u'男':1,u'女':0})
    data[u'体检日期'] = (pd.to_datetime(data[u'体检日期']) - parse('2017-10-09')).dt.days
 
    data.fillna(data.median(axis=0),inplace=True)

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat,test_feat



train_feat,test_feat = make_feat(train,test)

arr=[u'血糖',
# u'总胆固醇',
u'高密度脂蛋白胆固醇',
u'低密度脂蛋白胆固醇',
u'中性粒细胞%',
u'淋巴细胞%',
u'单核细胞%',
u'嗜酸细胞%',
u'嗜碱细胞%',
u'乙肝表面抗原',
u'乙肝表面抗体',
u'乙肝e抗原',
u'乙肝e抗体',
u'乙肝核心抗体',
]
train_feat[u'血红蛋白']=train_feat[u'血红蛋白']*train_feat[u'红细胞计数']
test_feat[u'血红蛋白']=test_feat[u'血红蛋白']*test_feat[u'红细胞计数']


predictors = [f for f in test_feat.columns if f not in arr]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)

print('begin...')
params = {
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
}

t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
# kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
kf = KFold(n_splits=5, shuffle=True,random_state=520)
for i, (train_index, test_index) in enumerate(kf.split(train_feat.index)):
    # if i==0:
    #     continue
    print(u'第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1[u'血糖'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2[u'血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=50)
    # if i==0:
    #     continue
    # feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    # print feat_imp
    # exit()
    # if i==0:
    #     train_preds[test_index] +=train_feat.loc[test_index,u'血糖']
    # else:
    train_preds[test_index] += gbm.predict(train_feat2[predictors])

    test_preds[:,i] = gbm.predict(test_feat[predictors])

print(u'线下得分：{}'.format(mean_squared_error(train_feat[u'血糖'],train_preds)*0.5))
print(u'CV训练用时{}秒'.format(time.time() - t0))
# print test_preds
 
# np.savetxt("./a.txt",test_preds,fmt='%.4f')
print test_preds.std()
test_preds=np.delete(test_preds,0,axis=1)
print test_preds.std()
# np.savetxt("./b.txt",test_preds,fmt='%.4f')
# print test_preds
submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
submission.to_csv(r'result{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None,
                  index=False, float_format='%.4f')