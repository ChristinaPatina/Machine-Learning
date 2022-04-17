import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pyexcel
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit, cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_validate
# from sklearn.cross_validation import KFold

import warnings
warnings.simplefilter('ignore')

filename = "electricity_plant.csv"
data_electricity = np.genfromtxt(filename, delimiter=',', skip_header=False)
data_electricity = data_electricity[1:]
# filename2 = "airfoil_self_noise.dat"
# data_airfoil = np.genfromtxt(filename2, dtype=[int, float, float, float, float, float], skip_header=False)
# print(data_airfoil)
# np.savetxt("data_airfoil.csv", data_airfoil, delimiter=',', fmt=['%d', '%.1f', '%.4f', '%.1f', '%.7f', '%.3f'])
filename3 = "data_airfoil.csv"
data_airfoil = np.genfromtxt(filename3, delimiter=',', skip_header=False)

# electricity
print("Наши данные Electricity:\n", data_electricity, "\n")
# X_electricity = data_electricity[:, 0]
# X_electricity = X_electricity.reshape(-1, 1)
X_electricity = data_electricity[:, 0:4]
# print("Наши X:\n", X_electricity, "\n")
Y_electricity = data_electricity[:, 4]
# print("Наши Y:\n", Y_electricity, "\n")

X_train, X_test, Y_train, Y_test = train_test_split(X_electricity, Y_electricity, test_size=0.33, random_state=42)
print("len X_electricity:", len(X_electricity))
print("len X_train:", len(X_train))
print("len X_test:", len(X_test), "\n")
# end electricity

# airfoil
#X_airfoil = data_airfoil[:, 0]
#X_airfoil = X_airfoil.reshape(-1, 1)
X_airfoil = data_airfoil[:, 0:5]
Y_airfoil = data_airfoil[:, 5]
# end airfoil

# Electricity
# Метод наименьших квадратов
#plt.figure()
# plt.plot(X_electricity[:, 3], Y_electricity, color='r', linewidth=4, label='y')
#plt.plot(X_electricity, Y_electricity, 'o', markersize=3, color='r')

# Create linear regression object
linear_model = LinearRegression()
# Train the model using the training sets
linear_model.fit(X_train, Y_train)
# Make predictions using the testing set
regression_y = linear_model.predict(X_test)

print('LinearRegression:')
print('Coefficient: ', linear_model.coef_)
print('Intercept: ', linear_model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(Y_test, regression_y))
print('R² Value: ', linear_model.score(X_test, Y_test))
print('Variance score: %.16f' % r2_score(Y_test, regression_y), "\n")

#plt.plot(X_electricity, regression_y, color='b', linewidth=2, label='predicted y')
#plt.legend(loc=4)
#plt.title('LinearRegression')
#plt.show()
# end Метод наименьших квадратов

# Ridge
ridge_model = Ridge(alpha=100)
ridge_model.fit(X_train, Y_train)
regression_y = ridge_model.predict(X_test)
print('Ridge:')
print('Coefficient:', ridge_model.coef_)
print('Intercept:', ridge_model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(Y_test, regression_y))
print('R² Value:', ridge_model.score(X_test, Y_test), '\n')
'''
n = 100001
coeficient_0 = []
coeficient_1 = []
coeficient_2 = []
coeficient_3 = []
coef_0 = []
coef_1 = []
coef_2 = []
coef_3 = []
X_alpha = list(range(0, n))
r2_value = []
for i in range(0, n):
    ridge_model = Ridge(alpha=i)
    ridge_model.fit(X_train, Y_train)
    regression_y = ridge_model.predict(X_test)
    coef_0, coef_1, coef_2, coef_3 = ridge_model.coef_
    coeficient_0.append(coef_0)
    coeficient_1.append(coef_1)
    coeficient_2.append(coef_2)
    coeficient_3.append(coef_3)
    r2_value.append(ridge_model.score(X_test, Y_test))
'''
'''plt.figure()
plt.title('coef-alpha')
plt.subplot(221)
plt.scatter(X_alpha, coeficient_0, linewidth=0.5)
plt.title('1 признак')
plt.subplot(222)
plt.scatter(X_alpha, coeficient_1, linewidth=2)
plt.title('2 признак')
plt.subplot(223)
plt.scatter(X_alpha, coeficient_2)
plt.title('3 признак')
plt.subplot(224)
plt.scatter(X_alpha, coeficient_3)
plt.title('4 признак')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()
'''
'''
plt.figure()
plt.plot(X_alpha, coeficient_0)
plt.title('Coef-alpha: 1 признак')
plt.show()
plt.figure()
plt.plot(X_alpha, coeficient_1)
plt.title('Coef-alpha: 2 признак')
plt.show()
plt.figure()
plt.plot(X_alpha, coeficient_2)
plt.title('Coef-alpha: 3 признак')
plt.show()
plt.figure()
plt.plot(X_alpha, coeficient_3)
plt.title('Coef-alpha: 4 признак')
plt.show()
plt.figure()
plt.plot(X_alpha, r2_value)
plt.title('R^2 score-alpha')
plt.show()
'''
# end Ridge

# Lasso
lasso_model = Lasso(alpha=10)
lasso_model.fit(X_train, Y_train)
regression_y = lasso_model.predict(X_test)
print('Lasso:')
print('Coefficient:', lasso_model.coef_)
print('Intercept:', lasso_model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(Y_test, regression_y))
print('R² Value:', lasso_model.score(X_test, Y_test), '\n')
'''
n = 1001
coeficient_0 = []
coeficient_1 = []
coeficient_2 = []
coeficient_3 = []
coef_0 = []
coef_1 = []
coef_2 = []
coef_3 = []
X_alpha = list(range(0, n))
r2_value = []
for i in range(0, n):
    lasso_model = Lasso(alpha=i)
    lasso_model.fit(X_train, Y_train)
    regression_y = lasso_model.predict(X_test)
    coef_0, coef_1, coef_2, coef_3 = lasso_model.coef_
    coeficient_0.append(coef_0)
    coeficient_1.append(coef_1)
    coeficient_2.append(coef_2)
    coeficient_3.append(coef_3)
    r2_value.append(lasso_model.score(X_test, Y_test))

plt.figure()
plt.plot(X_alpha, coeficient_0)
plt.title('Coef-alpha: 1 признак')
plt.show()
plt.figure()
plt.plot(X_alpha, coeficient_1)
plt.title('Coef-alpha: 2 признак')
plt.show()
plt.figure()
plt.plot(X_alpha, coeficient_2)
plt.title('Coef-alpha: 3 признак')
plt.show()
plt.figure()
plt.plot(X_alpha, coeficient_3)
plt.title('Coef-alpha: 4 признак')
plt.show()
plt.figure()
plt.plot(X_alpha, r2_value)
plt.title('R^2 score-alpha')
plt.show()
'''
# end Lasso
# end Electricity

# Airfoil
print("Наши данные Airfoil:\n", data_airfoil, "\n")

deg = 5
pol = PolynomialFeatures(degree=deg)
X_airfoil = pol.fit_transform(X_airfoil)

X_train, X_test, Y_train, Y_test = train_test_split(X_airfoil, Y_airfoil, test_size=0.33, random_state=42)
print("len X_airfoil:", len(X_airfoil))
print("len X_train:", len(X_train))
print("len X_test:", len(X_test), "\n")

# cross validation
'''
num_folds = 4
n_R = 100001
n_L = 1001

X_train_folds = np.array_split(X_train, num_folds)
Y_train_folds = np.array_split(Y_train, num_folds)

k_to_acc = {}
for k in range(n_R):
    k_to_acc[k] = list()

for k in range(n_R):
    for i in range(num_folds):
        X_val = X_train_folds[i]
        Y_val = Y_train_folds[i]
        X_tr = np.vstack((X_train_folds[0:i] + X_train_folds[i+1:]))
        # Y_tr = np.vstack((Y_train_folds[0:i] + Y_train_folds[i+1:]))
        Y_tr = Y_train_folds
        model = Ridge(alpha=k)
        ridge_model.fit(X_tr, Y_tr)
        regression_y = ridge_model.predict(X_val)

        sc = ridge_model.score(X_val, Y_val)
        k_to_acc[k].append(sc)
print("FINISHED")

best_k = 0

best_sc = 0

for k in range(n_R):
    sc = np.mean(k_to_acc[k])
    if sc > best_sc:
        best_sc = sc
        best_k = k
print("Best_alpha:", best_k)
print("Best_score:", best_sc)
'''


def score_model(estimator, X, y):
    n = 8
    cv = ShuffleSplit(n_splits=n, random_state=1)
    return cross_val_score(estimator, X=X, y=y, scoring='r2', cv=cv).mean()


linear_model = LinearRegression()
linear_model.fit(X_train, Y_train.tolist())
print("R^2 Score LinearRegression():", score_model(linear_model, X_test, Y_test.tolist()))
ridge_model = Ridge(alpha=5000)
ridge_model.fit(X_train, Y_train.tolist())
print("R^2 Score Ridge():", score_model(ridge_model, X_test, Y_test.tolist()))
'''
param_grid = {'alpha': [0.1, 10, 20.5, 11, 300, 598, 35]}
svm_gs = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                      param_grid, cv=10)
ridge_mod = svm_gs.fit(X_train, Y_train.tolist())
print("best param:", svm_gs.best_params_, "best score:", svm_gs.best_score_)
'''
lasso_model = Lasso(alpha=0.0001)
lasso_model.fit(X_train, Y_train)
print("R^2 Score Lasso():", score_model(lasso_model, X_test, Y_test))
param_grid = {'alpha': [0.1, 0.0001, 0.01, 10, 26, 11, 44, 5, 12]}
svm_gs = GridSearchCV(Lasso(),
                      param_grid, cv=ShuffleSplit(n_splits=8, random_state=1))
ridge_mod = svm_gs.fit(X_train, Y_train.tolist())
print("best param:", svm_gs.best_params_, "best score:", svm_gs.best_score_)

'''
for n in range(0, 50):
    lasso_model = Lasso(alpha=n)
    linear_model.fit(X_train, Y_train)
    print(n, score_model(lasso_model, X_test, Y_test))
'''
# Set the parameters by cross-validation

# end gr search

# Метод наименьших квадратов
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
regression_y = linear_model.predict(X_test)

print('LinearRegression:')
print('Coefficient: ', linear_model.coef_)
print('Intercept: ', linear_model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(Y_test, regression_y))
print('R² Value: ', linear_model.score(X_test, Y_test), "\n")
# end Метод наименьших квадратов

# Ridge
ridge_model = Ridge(alpha=100)
ridge_model.fit(X_train, Y_train)
regression_y = ridge_model.predict(X_test)
print('Ridge:')
print('Coefficient:', ridge_model.coef_)
print('Intercept:', ridge_model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(Y_test, regression_y))
print('R² Value:', ridge_model.score(X_test, Y_test), '\n')
'''
n = 501
X_alpha = list(range(0, n))
r2_value = []
for i in range(0, n):
    ridge_model = Ridge(alpha=i)
    ridge_model.fit(X_train, Y_train)
    regression_y = ridge_model.predict(X_test)
    r2_value.append(ridge_model.score(X_airfoil, Y_airfoil))
plt.figure()
plt.plot(X_alpha, r2_value)
plt.title('R^2 score-alpha')
plt.show()
'''
# end Ridge

# Lasso
lasso_model = Lasso(alpha=0.0001)
lasso_model.fit(X_train, Y_train)
regression_y = lasso_model.predict(X_test)
print('Lasso:')
print('Coefficient:', lasso_model.coef_)
print('Intercept:', lasso_model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(Y_test, regression_y))
print('R² Value:', lasso_model.score(X_test, Y_test), '\n')
'''
n = 101
X_alpha = list(range(0, n))
r2_value = []
for i in range(0, n):
    lasso_model = Lasso(alpha=i)
    lasso_model.fit(X_train, Y_train)
    regression_y = lasso_model.predict(X_test)
    r2_value.append(lasso_model.score(X_test, Y_test))
plt.figure()
plt.plot(X_alpha, r2_value)
plt.title('R^2 score-alpha')
plt.show()
'''
# end Lasso
# end Airfoil

