# coding=utf-8
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from columns import *

test="../data/d_test_A_20180102.csv"
train="../data/d_train_20180102.csv"

data=pd.read_csv(train, encoding="gb2312")
# columns_en.remove("blood sugar")
data.columns=columns_en
data = data.replace("??",u'女')
data = data.replace({u'男',u'女'},{1,0})
data["gender"]=data["gender"].astype("int")
# data["Physical examination date"]=pd.to_datetime(data["Physical examination date"])

# Y = data['blood sugar']
drop=['id', 'Physical examination date'] 
data = data.drop(drop, 1)
for v in drop:
    columns_en.remove(v)
# data["Physical examination date"]=pd.to_datetime(data["Physical examination date"])
# a=data["Physical examination date"].value_counts()

threhold=(6.0-data["blood sugar"].min())/(data["blood sugar"].max()-data["blood sugar"].min())
# print data["blood sugar"].max(),data["blood sugar"].min()
data=(data-data.min())/(data.max()-data.min())

# print threhold
# exit()
df_norm=data.loc[data['blood sugar']<=threhold]
df_unno=data.loc[data['blood sugar']>threhold]
m1=df_norm.mean()
m2=df_unno.mean()
c = pd.DataFrame([m1,m2])
# c=(c-c.min())/(c.max()-c.min())
r=m2-m1
# print r
print r[abs(r)>0.02].index
print len(r[abs(r)>0.02].index)
# print c
# c.T.plot()
# plt.show()
# # print df["id"].count()
# exit()
# grouped = df['blood sugar'].groupby(df['Physical examination date'])
# # print a.sort_index()
# print grouped.mean()
# print grouped.min()
# print grouped.max()
# print type(grouped)
# print a.sort_index(by="Physical examination date")["Physical examination date"]

# X=(X-X.min())/(X.max()-X.min())
# Y=(Y-Y.min())/(Y.max()-Y.min())
# res={}
# print columns_en[10]
# a=np.array([X[columns_en[10]].values,Y.values])
# print a.shape
# for v in X.columns:
#     print np.cov(np.array([X[v].values,Y.values]))
# print res



# dataLenth = len(columns_en)
# data=X.loc[0].values
# angles = np.linspace(0, 2*np.pi, dataLenth, endpoint=False) 
# data = np.concatenate((data, [data[0]]))
# angles = np.concatenate((angles, [angles[0]]))

# fig = plt.figure() 
# ax = fig.add_subplot(111, polar=True) 
# ax.plot(angles, data, 'bo-', linewidth=2) 
# ax.set_thetagrids(angles * 180/np.pi, range(len(columns_en)), fontproperties="SimHei") 
# ax.set_title("matplotlib radar", va="bottom", fontproperties="SimHei") 
# ax.grid(True) 
# plt.show()

# print X.head()
# a=X.values.T
# # a=X[columns_en].values.T
# a=np.cov(a)
# print a.shape
# plt.imshow(a)
# plt.show()

# X=X.fillna(-1)

# print X.values.shape
# plt.im()
# idx=1000
# plt.imshow(X.values[idx:idx+100,:])
# plt.show()
# a=data["High density lipoprotein cholesterol"].drop(range(18),0).values.reshape([74,76])
# # print 74*76
# # print a.shape
# b=data["Hepatitis B core antibody"].drop(range(18),0).values.reshape([74,76])

# fg=plt.figure(0)
# plt.imshow(a)
# fg=plt.figure(1)
# plt.imshow(b)
# plt.show()

# # print a.shape
# def analyst(data):
#     print "  [",data.count(),data.min(),data.max(),data.isnull().sum(),"]"

# # print X[columns_en[20:30]].head()
# # a=X["Hepatitis B core antibody"].head()
# # print a
# # print a[0]==np.nan

# for v in columns_en:
#     if X[v].count()>5000:
#         print "'%s',"%v
    # analyst(X[v])

# print columns_en[0:3]
# print columns_en[4:20]
# print X[columns_en[4:41]].head()
# print X.head()

# print data.mean()
# print data.columns
# print data