import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
# print X
y = np.sin(X).ravel()


# exit()

# #############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
print y_rbf
# y_lin = svr_lin.fit(X, y).predict(X)
# y_poly = svr_poly.fit(X, y).predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
# plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
# plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# from sklearn import datasets
# import numpy as np
# from sklearn.cross_validation import train_test_split

# iris = datasets.load_iris()
# X = iris.data[:, [2, 3]]
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # from sklearn.svm import SVR
# # clf=SVR()
# from 

# X=np.array([[1],[2],[3],[4],[5],[6],[7]],dtype=np.float)
# y=np.array([1,2,3,4,5,6,7],dtype=np.float)
# test=np.array([[8],[9],[3],[4],[7]],dtype=np.float)

# clf.fit(X, y)
# a=clf.predict(test)
# print a
# # print sum(abs(a-y_test))
# # a=lr.predict_proba(test)

# # print iris
# # from sklearn.preprocessing import StandardScaler
# # sc = StandardScaler()
# # sc.fit(X_train)
# # X_train_std = sc.transform(X_train)
# # X_test_std = sc.transform(X_test)

# # X_combined_std = np.vstack((X_train_std, X_test_std))
# # y_combined = np.hstack((y_train, y_test))
# # X=np.array([[1],[2],[3],[4],[5],[6],[7]],dtype=np.float)
# # y=np.array([1,2,3,4,5,6,7],dtype=np.float)
# # test=np.array([[8],[9],[3],[4],[7]],dtype=np.float)
# # from sklearn.linear_model import LogisticRegression
# # lr = LogisticRegression(C=100.0, random_state=0)
# # # print y_train.shape

# # lr.fit(X_train, y_train)
# # a=lr.predict(X_test)
# # print sum(abs(a-y_test))
# # # a=lr.predict_proba(test)
# # print a
# # print y_test
# # plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
# # plt.xlabel('petal length [standardized]')
# # plt.ylabel('petal width [standardized]')
# # plt.legend(loc='upper left')
# # plt.show()