# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
import numpy as np


# kf = KFold(100, n_folds = 5, shuffle=True, random_state=520)
# for i, (train_index, test_index) in enumerate(kf):
#     print i
#     print train_index,test_index

kf = KFold(n_splits=5, shuffle=True,random_state=np.random.randint(11))
for (i,(t,tt)) in enumerate(kf.split(range(100))):
    # print kf._iter_test_indices()
    # print test_index
    print i
    print tt
# print dir(kf)
