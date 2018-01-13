# coding=utf-8
import numpy as np 
import pandas as pd
file_small=[
    "predicts.csv",
    "offline_train.csv",
    "online_train.csv",
]
# dates=pd.date_range('20180101',periods=10)
# print dates
# print type(dates)

# df=pd.DataFrame({"A":["20160101","20160102"]})
# df["A"]=pd.to_datetime(df["A"])
# print df
# print df.dtypes
# exit()
path='../'
# for v in file_small:
#     data = pd.read_csv(path+v)
#     data.fillna(0)
#     print path+v
#     print data.columns
#     print data.dtypes
file=path+file_small[2]
data = pd.read_csv(file)
data=data.replace('null',-1)
# data["Distance"]=data["Distance"].astype('int')
data.columns=data.columns.str.lower()
for v in data.columns:
    print v
print data.dtypes
# print data["Discount_rate"]
   


    # print path+v
    # print type(predicts)
# key=["User_id","Merchant_id","Coupon_id","Date_received"]
# # for v in key:
# #     print v
# #     print predicts[v].count()
# #     print predicts[v].value_counts().count()
# a=predicts.loc[0,:]
# b=a.values
# print b
# print type(b)
# # print b[0]
# for v in b:
#     print type(v)

# print "ok"