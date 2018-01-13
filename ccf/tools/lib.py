import numpy as np 
import pandas as pd

def merchant(data):
    c=data["merchant_id"]
    # print c.value_counts().sum()
    print c[c<1].count()
# merchant(data)

def coupon(data):
    c=data["coupon_id"]
    print c[c=="null"].count()
# coupon(data)

def analyst_data(data):
    data["date_received"]=data["date_received"].replace("null",pd.NaT)
    data["date_received"]=pd.to_datetime(data["date_received"])
    dates=predicts["date_received"]
    print dates.min()
    print dates.max()
 # analyst_data(data)
def analyst_num(data):
    print data.count()
# analyst_num(data)

def analyst_user(data):
    print data["user_id"].nunique()
    print data["merchant_id"].nunique()
    print data["coupon_id"].nunique()
# analyst_user(data)
def analyst_discount(data):
    d= data["discount_rate"]
    t=d.value_counts()
    print t.count()
    s=pd.Series(t.index)
    print s
    print s[s.str.contains(":")].count()
    print s[s.str.contains("\.")].count()
    print s[s.str.contains("null")].count()
# analyst_discount(data)

def analyst_distance(data):
    print data["distance"].value_counts()
# analyst_distance(data)

def analyst_date_b(data):
    data["date"]=data["date"].replace("null",pd.NaT)
    data["date"]=pd.to_datetime(data["date"])
    dates=data["date"]
    print dates.min()
    print dates.max()
# analyst_date_b(data)

def analyst_action(data):
    d= data["action"]
    t=d.value_counts()
    print t
    print t.count()
    s=pd.Series(t.index)
    print s