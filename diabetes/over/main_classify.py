# coding=utf-8
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import time

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from tools.columns import *

test_path="./data/d_test_A_20180102.csv"
train="./data/d_train_20180102.csv"

data=pd.read_csv(train, encoding="gb2312")
data.columns=columns_en
data = data.replace("??",u'女')
data = data.replace({u'男',u'女'},{1,0})
data["gender"]=data["gender"].astype("int")
data=data.fillna(data.mean())

blood_threhold=6.2
ill=pd.Series([0. for i in range(len(data))])
ill[data['blood sugar']>blood_threhold]=1.
classifyx=data[classify_columns]
classify=ill

normal=data[data['blood sugar']<=blood_threhold]
unnormal=data[data['blood sugar']>blood_threhold]
choose=columns_en[:]
choose.remove("blood sugar")
choose.remove("Physical examination date")
choose.remove("id")


normal_x=normal[choose]
normal_y=normal['blood sugar']
classf=["Hemoglobin","Uric acid"]
unnormal_x=unnormal[classf]
unnormal_y=unnormal['blood sugar']

classifyx=data[classf]
classify=ill

clsx=["Uric acid","blood sugar"]
a=(data[clsx]-data[clsx].min())/(data[clsx].max()-data[clsx].min())
a.loc[range(0,5000,50),:].plot()
plt.show()
# def classfy_func(X,Y,pca_num=9,layer=(50,10),alpha=1e-5):
#     X.index=range(len(X))
#     Y.index=range(len(Y))
#     X=(X-X.min())/(X.max()-X.min())

#     kf = KFold(n_splits=3, shuffle=True,random_state=np.random.randint(11))
#     accuracy=0.0
#     startTime = time.clock()
#     err=[]
#     for train_index, test_index in kf.split(X.index):
#         train=X.loc[train_index].values
#         trainy=Y.loc[train_index].values.ravel()
#         test=X.loc[test_index].values
#         testy=Y.loc[test_index].values.ravel()

#         # print trainy[0:10]
#         # print testy[0:10]
#         # pca = PCA(n_components=pca_num)
#         # pca.fit(train)
#         # train=pca.transform(train)
#         # test=pca.transform(test)

#         # clf = MLPRegressor(alpha=alpha, hidden_layer_sizes=layer, random_state=1)
#         clf = MLPClassifier(alpha=alpha, hidden_layer_sizes=layer, random_state=1)
#         clf.fit(train, trainy) 
#         predy = clf.predict(test)
#         print 1-sum(abs(predy-testy)/float(len(predy)))
#         # print testy[0:10]
#         # exit()

#         # accy_tmp=sum((testy-predy)**2)/float(2*len(testy))
#         # err.append(accy_tmp)
#         # print accy_tmp

#     #     plt.plot(range(10),predy[0:10],c='r')
#     #     plt.plot(range(10),testy[0:10],c='g')
#     #     plt.show()
#     #     exit(0)

#     # runningTime = float(time.clock() - startTime)
#     # err=np.array(err)
#     # strx="final(%d,%d,%d,%f)(%f):%f %f \n"%(pca_num,layer[0],layer[1],alpha,
#     #     runningTime,err.mean(),err.std())
#     # print strx
#     # with open("./data/result.txt",'a') as f:
#     #     f.write(strx)
   
# classfy_func(classifyx,classify,pca_num=9,layer=(50,10),alpha=1e-5)

# print unnormal.head()

# data.loc[data['gender']<1,"age"].T.hist(bins=90)
# data.loc[data['gender']>0,"age"].T.hist(bins=90)

# data['gender'].T.hist()
# allx=data[data['blood sugar']<15]
# grouped = data['blood sugar'].groupby(data['age'])
# print grouped
# print type(grouped)
# print grouped.get_group(38)
# data=data.drop(["Physical examination date","id"],axis=1)
# norm=[22,673,4695]
# # norm=[22,673,4695]
# unno=[127,416,1707]


# com=[22,673,416,1707]
# # print data.loc[norm,:].T
# data.loc[com,classify_columns].T.plot()
# # data.loc[unno,:].T.plot()
# plt.show()

# for k,v in grouped:
#     print k,v
# print grouped["35"]
# print grouped.count()
# # print grouped.min()
# # print grouped.max()
# print grouped.mean()
# print grouped.std()
# grouped.T.hist()
# woman=data.loc[data['gender']<1,"blood sugar"].T.hist(bins=60)
# man=data.loc[data['gender']>0,"blood sugar"].T.hist(bins=60)
# .T.hist(bins=60)
# plt.show()
# exit()
# choose=columns_en[:]
# # choose.remove("blood sugar")
# choose.remove("Physical examination date")
# choose.remove("id")
# unnormal=unnormal[choose]
# unnormal=(unnormal-unnormal.min())/(unnormal.max()-unnormal.min())
# val=unnormal.values.T
# # print val.shape
# cov=np.cov(val)
# np.set_printoptions(formatter={'float':lambda x: "%5.5f"%float(x)})
# cov=abs(cov)
# # print cov
# cov=cov/cov.sum(axis=0)
# # exit()
# final=cov[:,-1]
# print final
# # print choose[final>0.025]
# exit()
# plt.imshow(cov)
# plt.show()
# exit()

# b=data['blood sugar']
# print data["id"].count()
# print b.describe()
# result=""
# for v in b:
#     result+="%.3f\n"%v
# with open("./data/tmp_show.txt",'w') as f:
#     f.write(result)
# print "ok"
# exit()

def getresult(pca_num=9,layer=(50,10),alpha=1e-5):
    # global X,Y
    global classifyx,classify

    test=pd.read_csv(test_path, encoding="gb2312")
    # print test.index
    columns_en.remove('blood sugar')
    test.columns=columns_en
    test = test.replace("??",u'女')
    test = test.replace({u'男',u'女'},{1,0})
    test["gender"]=test["gender"].astype("int")
    test=test.fillna(test.mean())
    test_data = test[choose]
    # test_data=(test_data-test_data.min())/(test_data.max()-test_data.min())
    # classify_columns.remove('blood sugar')
    test_c=test[classify_columns]
    clf =  MLPClassifier(hidden_layer_sizes=(20,20))
    clf.fit(classifyx, classify) 
    predc = clf.predict(test_c)
    print predc
    # print np.where(predc>0)
    ill=test_c[predc>0]
    normal=test_c[predc<1]
    # print ill.count()
    # print normal.count()


    # train=X.values
    # trainy=Y.values.ravel()
    # test_data=test_data.values
    # np.set_printoptions(formatter={'float':lambda x: "%5.3f"%float(x)})

    # pca = PCA(n_components=pca_num)
    # pca.fit(X)
    # X=pca.transform(X)
    # test_data=pca.transform(test_data)

    # clf = MLPRegressor(alpha=alpha, hidden_layer_sizes=layer, random_state=1)
    # clf.fit(X, Y) 
    # predy = clf.predict(test_data)
    # result=""
    # for v in predy:
    #     result+="%.3f\n"%v

    # with open("./data/predict_result.txt",'w') as f:
    #     f.write(result)
    # print "ok"

def execute(X,Y,pca_num=9,layer=(50,10),alpha=1e-5):
    X.index=range(len(X))
    Y.index=range(len(Y))
    X=(X-X.min())/(X.max()-X.min())

    kf = KFold(n_splits=3, shuffle=True,random_state=np.random.randint(11))
    accuracy=0.0
    startTime = time.clock()
    err=[]
    for train_index, test_index in kf.split(X.index):
        train=X.loc[train_index].values
        trainy=Y.loc[train_index].values.ravel()
        test=X.loc[test_index].values
        testy=Y.loc[test_index].values.ravel()

        # print trainy[0:10]
        # print testy[0:10]
        # pca = PCA(n_components=pca_num)
        # pca.fit(train)
        # train=pca.transform(train)
        # test=pca.transform(test)

        clf = MLPRegressor(alpha=alpha, hidden_layer_sizes=layer, random_state=1)
        clf.fit(train, trainy) 
        predy = clf.predict(test)
        # print predy[0:10]

        accy_tmp=sum((testy-predy)**2)/float(2*len(testy))
        err.append(accy_tmp)
        print accy_tmp

        plt.plot(range(10),predy[0:10],c='r')
        plt.plot(range(10),testy[0:10],c='g')
        plt.show()
        exit(0)

    runningTime = float(time.clock() - startTime)
    err=np.array(err)
    strx="final(%d,%d,%d,%f)(%f):%f %f \n"%(pca_num,layer[0],layer[1],alpha,
        runningTime,err.mean(),err.std())
    print strx
    with open("./data/result.txt",'a') as f:
        f.write(strx)
    
# if __name__=="__main__":
#     # print normal_x.describe()
#     # execute(normal_x,normal_y,pca_num=7,layer=(20,40),alpha=1e-5)
#     execute(unnormal_x,unnormal_y,pca_num=7,layer=(50,40),alpha=10)
#     # getresult(pca_num=9,layer=(20,40),alpha=1e-5)
#     print "ok"