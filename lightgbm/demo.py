# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error


# load or create your dataset
print('Load data...')
# df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
# df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')

path="./data/"
df=pd.read_csv(path+"test.csv")
df=df.fillna(df.mean())
df = df.replace("??",'女')
df = df.replace({'男','女'},{1,0})
df["性别"]=df["性别"].astype("int")
ta=range(0,5000,100)
tb=range(1,5001,500)

y_train = df["血糖"].loc[ta].values
y_test = df['血糖'].loc[tb].values
columns=["id", '体检日期', '血糖']
# need=["白球比例","甘油三酯","总胆固醇"]
X_train = df.drop(columns, axis=1).loc[ta,:].values
X_test = df.drop(columns, axis=1).loc[tb,:].values


# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print y_pred
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)