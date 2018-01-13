import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s=pd.Series([1,3,4,np.nan,6,8])
dates=pd.date_range('20180108',periods=6)
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list("ABCD"))

df2=pd.DataFrame({
        'A':1.,
        'B':pd.Timestamp('20180108'),
        'C':pd.Series(1,index=list(range(4)),dtype='float32'),
        'D':np.array([3]*4,dtype='int32'),
        'E':pd.Categorical(["test","train","test","train"]),
        'F':"foo",
})

df=pd.DataFrame({"A":["20160101", "20160102", "20160103","null"],"B":[1, 2, 3,5]})
df=df.replace("null",pd.NaT)

# print np.nat
df["A"]=pd.to_datetime(df["A"])
print df
# print type(df.loc[3,"A"])
# print df[""]

# for v in pd.Series([1, 2, 3]):
#     print v

# df=pd.DataFrame({"A":[1, 2, 3],"B":[1, 2, 3]})
# for v in df:
#     for vv in df[v]:
#         print vv



# s=pd.Series(np.random.randn(1000),index=pd.date_range("20180101",periods=1000))
# s=s.cumsum()
# s.plot()
# plt.show()

# df=pd.DataFrame(np.random.randn(1000,4),
#     index=pd.date_range("20180101",periods=1000),columns=list("ABCD"))
# df=df.cumsum()
# df.plot()
# plt.show()

# df = pd.DataFrame(
#     {"id":[1,2,3,4,5,6],
#         "raw_grade":['a', 'b', 'b', 'a', 'a','e']})
# df["grade"]=df["raw_grade"].astype("category")
# df["grade"].cat.categories=["very good","good","very bad"]
# # df["grade"].cat.categories=["very bad","very good","good"]
# print df.sort_values(by="grade")
# print df.groupby("grade").size()

# dates=pd.date_range('2018-01-01',periods=10,freq='M')
# s=pd.Series(np.random.randint(0,500,len(dates)),index=dates)
# a=s.to_period()
# b=a.to_timestamp()
# print b

# # print s
# print s.resample('1Min').sum()
# print dates

# df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
#     'B' : ['A', 'B', 'C'] * 4,
#     'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
#     'D' : np.random.randn(12),
#     'E' : np.random.randn(12)})
# print df
# pivot=pd.pivot_table(df,values="D",index=["A","B"],columns=["C"])
# pivot=pd.pivot_table(df,values="D",index=["A"],columns=["C"])
# print pivot

# print df2
# stack=df2.stack()
# print stack
# print stack.unstack(1)
# print stack.unstack(0)

# tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
#     'foo', 'foo', 'qux', 'qux'],
#     ['one', 'two', 'one', 'two',
#     'one', 'two', 'one', 'two']]))
# index=pd.MultiIndex.from_tuples(tuples,names=['first','second'])
# df=pd.DataFrame(np.random.randn(8,2),index=index,columns=list("AB"))
# print tuples
# print index
# print df

# df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
#     'foo', 'bar', 'foo', 'foo'],
#     'B' : ['one', 'one', 'two', 'three',
#     'two', 'two', 'one', 'three'],
#     'C' : np.random.randn(8),
#     'D' : np.random.randn(8)}
# )
# print df
# print df.groupby(["A","B"]).count()

# df=pd.DataFrame(np.random.randn(8,4),columns=list("ABCD"))
# print df
# s=df.iloc[3]
# df=df.append(s,ignore_index=True)
# print df


# left=pd.DataFrame({"key":["foo","too"],"lval":[1,2]})
# right=pd.DataFrame({"key":["foo","too"],"rval":[4,5]})
# print left
# print right
# print pd.merge(left,right,on='key')



# df=pd.DataFrame(np.random.randn(10,4))
# print df
# pieces=[df[:3],df[3:7],df[7:]]
# print pieces
# print pd.concat(pieces)


# df.columns= df.columns.str.lower()
# print df
# print type(df.columns)
# print type(s)

# s=pd.Series(['A', 'B', 'C', 'Aaba', 
#     'Baca', np.nan, 'CABA', 'dog', 'cat'])
# print s.str.lower()

# s = pd.Series(np.random.randint(0, 7, size=10))
# print s
# print s.value_counts()


# df["E"]=[1,2,3,4,5,6]
# print df
# print df.apply(np.cumsum)
# print df.apply(lambda x:x.max())
# print df.max(1)

# print df 
# s1=pd.Series([1,2,3,np.nan,6,8],index=dates).shift(2)
# print s1
# print df.sub(s1,axis='index')

# print s1
# print df.mean()
# print df.mean(1)

# df1=df.reindex(index=dates[0:4],columns=list(df.columns)+['E'])
# df1.loc[dates[0]:dates[1],'E']=1
# df1=df1.dropna(how='any')
# print df1
# df1=df1.fillna(value=5)
# print df1
# print df1.isnull()
# print df1



# print df
# df2=df.copy()
# df2[df2>0]=-df2
# print df2

# s1=pd.Series([1,2,3,4,5,6],index=dates)
# df["F"]=s1
# df.at[dates[0],'A']=0
# df.iat[0,1]=0
# df.loc[:,'D']=np.array([5]*len(df))
# print df



# df3=df.copy()
# df3["E"]=['one', 'one','two','three','four','three']
# print df3[df3["E"].isin(["two","four"])]
# print df[df["D"].isin([])]
# print df[df>0]
# df3=df.copy()
# df3=df
# df3.columns=["1a",'2b','3c','4d']
# print df3
# print df


# print df.iloc[2]
# print df.iloc[2:4,1:3]
# print df.iloc[[2,4],[1,3]]
# print df.iloc[[1,2],:]
# print df.iloc[1,2]
# print df.iat[1,2]

# print df.loc[dates[0]]
# print df.loc[:,["A","B"]]
# print df.loc['20180109':'20180111',["A","B"]]
# print df.at[dates[0],"A"]

# print df[0:1]
# print df['A']
# print df[0:3]
# print df['20180109':'20180111']

# print df2.dtypes
# for v in df2.dtypes:
#     print type(v)
# print type(df2.dtypes)
# print df2.head(2)
# print df2.tail(2)
# print df2.columns
# print df2.index
# print df2.values
# print df.describe()
# print df.T
# print df
# print df.sort_index(axis=0,ascending=True)
# print df.sort_values(by="B",axis=0,ascending=False)