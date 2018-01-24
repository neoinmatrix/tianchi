# coding=utf-8
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import time

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
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
Y = data['blood sugar']
choose=columns_en[:]

remove=['id','blood sugar','Physical examination date']
for v in remove:
    choose.remove(v)

# X = data[choose]
X = data[classify_columns]

# print len(choose)
X=(X-X.min())/(X.max()-X.min())
# Y=(Y-Y.min())/(Y.max()-Y.min())

def getresult(pca_num=9,layer=(50,10),alpha=1e-5):
    global X,Y
    test=pd.read_csv(test_path, encoding="gb2312")
    # print test.index
    columns_en.remove('blood sugar')
    test.columns=columns_en
    test = test.replace("??",u'女')
    test = test.replace({u'男',u'女'},{1,0})
    test["gender"]=test["gender"].astype("int")
    # test_data = test[choose]
    test_data = test[classify_columns]
    test_data=test_data.fillna(test_data.mean())
    test_data=(test_data-test_data.min())/(test_data.max()-test_data.min())
   

    train=X.values
    trainy=Y.values.ravel()
    test_data=test_data.values
    # np.set_printoptions(formatter={'float':lambda x: "%5.3f"%float(x)})

    # pca = PCA(n_components=pca_num)
    # pca.fit(X)
    # X=pca.transform(X)
    # test_data=pca.transform(test_data)

    clf = MLPRegressor(alpha=alpha, hidden_layer_sizes=layer, random_state=1)
    clf.fit(X, Y) 
    predy = clf.predict(test_data)
    result=""
    for v in predy:
        result+="%.3f\n"%v

    with open("./data/predict_result20180111.csv",'w') as f:
        f.write(result)
    print "ok"



def execute(pca_num=9,layer=(50,10),alpha=1e-5):
    kf = KFold(n_splits=9, shuffle=True,random_state=np.random.randint(11))
    accuracy=0.0
    startTime = time.clock()
    err=[]
    for train_index, test_index in kf.split(X.index):
        train=X.loc[train_index].values
        trainy=Y.loc[train_index].values.ravel()
        test=X.loc[test_index].values
        testy=Y.loc[test_index].values.ravel()

        # pca = PCA(n_components=pca_num)
        # pca.fit(train)
        # train=pca.transform(train)
        # test=pca.transform(test)

        clf = MLPRegressor(alpha=alpha, hidden_layer_sizes=layer, random_state=1)
        clf.fit(train, trainy) 
        predy = clf.predict(test)

        accy_tmp=sum((testy-predy)**2)/float(2*len(testy))
        # accuracy+=accy_tmp
        err.append(accy_tmp)
        print accy_tmp
    runningTime = float(time.clock() - startTime)
    err=np.array(err)
    strx="final(%d,%d,%d,%f)(%f):%f %f \n"%(pca_num,layer[0],layer[1],alpha,
        runningTime,err.mean(),err.std())
    print strx
    with open("./data/result.txt",'a') as f:
        f.write(strx)
    

if __name__=="__main__":
    # execute(15,(20,40))
    getresult(15,(20,40))
    # df=pd.read_csv('./data/predict_result.txt',header=None)
    # print df.describe()
    # result=""
    # for v in Y:
    #     result+="%.3f\n"%v

    # with open("predict_result2.txt",'w') as f:
    #     f.write(result)
    print "ok"