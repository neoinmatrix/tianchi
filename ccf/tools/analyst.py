# coding=utf-8
import numpy as np 
import pandas as pd
from lib import * 

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
path="../"+file_arr[1]
data = pd.read_csv(path)
data.columns=data.columns.str.lower()
# data=data.replace("null",np.nan)

def action(data):
    df=data[data["coupon_id"]!="null"]
    df=df[df["distance"]!="null"]
    # print df.head()
    print df["distance"].count()
    df=df[df["date"]=="null"]
    print df["date"].count()

    # df=data[data["coupon_id"]!="null"]
    # # print df.head()
    # print df["date"].count()

    # dfa=df[df["date"]=="null"]
    # # print dfa.head()
    # print dfa["date"].count()

    # dfb=df[df["date"]!="null"]
    # # print dfb.head()
    # print dfb["date"].count()
    

    # df=data[data["coupon_id"]=="null"]
    # print df.head()
    # print df.count()
    # df=data[data["action"]==0]
    # print df.head()
    # print df.count()
    # df=data[data["action"]==2]
    # print df.head()
    # print df.count()

    # analyst_discount(df)
action(data)


# analyst_action(data)

# print dates.count_values()
# for r,s in zip(file_arr,file_small):
#     smallfile(r,s)
# print "ok"

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


# exit()
# print "offline:"
# data = pd.read_csv("./ccf_offline_stage1_train.csv") 
# key=["User_id","Merchant_id","Coupon_id","Date_received"]
# for v in key:
#     print v
#     print data[v].count()
#     print data[v].value_counts().count()

# print "online:"
# data = pd.read_csv("./ccf_oonline_stage1_train.csv") 
# key=["User_id","Merchant_id","Coupon_id","Date_received"]
# for v in key:
#     print v
#     print data[v].count()
#     print data[v].value_counts().count()
