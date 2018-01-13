# coding=utf-8
import json
import lightgbm as lgb
import pandas as pd
# from sklearn.metrics import roc_auc_score
# path="/Users/shuubiasahi/Documents/githup/LightGBM/examples/regression/"
# print("load data")
path="./"
df=pd.read_csv(path+"test.csv")

df=df.fillna(df.mean())

df = df.replace("??",'女')
df = df.replace({'男','女'},{1,0})
df["性别"]=df["性别"].astype("int")

ta=range(0,5000,10)
tb=range(1,5001,50)

y_train = df["血糖"].loc[ta].values
y_test = df['血糖'].loc[tb].values
columns=["id", '体检日期', '血糖']
X_train = df.drop(columns, axis=1).loc[ta,:].values
X_test = df.drop(columns, axis=1).loc[tb,:].values


# print y_train
# print len(y_train)
# print X_train
# print X_train.shape
# print X_test.shape

# print y_train
# print y_test

# exit()

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    # 'objective': 'binary',
    'metric': {'l1','l2', 'auc'},
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
gbm.save_model(path+'model.txt')
print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print(y_pred)
print y_test
accy_tmp=sum((y_pred-y_test)**2)/float(2*len(y_test))
print accy_tmp

# print('The roc of prediction is:', roc_auc_score(y_test, y_pred) )
print('Dump model to JSON...')
# dump model to json (and save to file)
model_json = gbm.dump_model()
with open(path+'model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

print('Feature names:', gbm.feature_name())
print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))