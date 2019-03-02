# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 02:46:14 2019

@author: PITAHAYA
"""

#%% package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

#%% data import and cleaning
df = pd.read_csv('loan-training.csv', header=0, index_col=0)
print(df.head())
print(df.info())

# fill NA values
df.fillna(df.mean(), inplace=True)

# test-train split
X_tr = df.iloc[0:int(0.9*len(df))]
y_tr = X_tr[df.columns[0]]
X_tr = X_tr[df.columns[1:]]

X_te = df.iloc[int(0.9*len(df)):]
y_te = X_te[df.columns[0]]
X_te = X_te[df.columns[1:]]

# transform data to have mean=0, SD=1. We only want to use data from X_tr.
SS = StandardScaler()
X_tr = SS.fit_transform(X_tr)
X_te = SS.transform(X_te) # we did not use any information from X_te.
print(np.mean(X_tr), np.std(X_tr))
print(np.mean(X_te), np.std(X_te))

#%% Linear Regression model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_tr, y_tr) # train the regressor
y_predlin = LR.predict(X_te) # make predictions
print(np.sum(np.round(y_predlin) == y_te.values) / len(y_te))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_te, np.round(y_predlin)))

#%% Logistic Regression model
from sklearn.linear_model import LogisticRegression
LoR = LogisticRegression()
LoR.fit(X_tr, y_tr) # train the classifier
y_predlog = LoR.predict_proba(X_te).T[1] # make predictions, get probability
print(np.sum(np.round(y_predlog) == y_te.values) / len(y_te))
print(roc_auc_score(y_te, y_predlog))

#%% CV: there's no need to manually split into train-test anymore
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
X = df[df.columns[1:]]
y = df[df.columns[0]]
clf = make_pipeline(StandardScaler(), LogisticRegression())
print(np.mean(cross_val_score(clf, X, y, cv=8, scoring='roc_auc', verbose=1, n_jobs=4)))

# does data standardization really matter?
print(np.mean(cross_val_score(LogisticRegression(), X, y, cv=8, scoring='roc_auc', verbose=1, n_jobs=4)))

# try a different regularization hyperparameter
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1e-1))
print(np.mean(cross_val_score(clf, X, y, cv=8, scoring='roc_auc', verbose=1, n_jobs=4)))

#%% XGBoost
import xgboost as xgb
clf = make_pipeline(StandardScaler(), xgb.XGBClassifier())
print(np.mean(cross_val_score(clf, X, y, cv=8, scoring='roc_auc', verbose=1, n_jobs=4)))

# try without standardization: I did tell you XGBoost is robust to scale
# XGBoost is also robust against multicollinearity
clf = xgb.XGBClassifier()
print(np.mean(cross_val_score(clf, X, y, cv=8, scoring='roc_auc', verbose=1, n_jobs=4)))

# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
# so many hyperparameters, what to do?

#%% Bayesian Optimization
from bayes_opt import BayesianOptimization
from datetime import datetime
def xgb_eval(max_depth, learning_rate, n_estimators,
             gamma, reg_alpha, reg_lambda):
    print(datetime.now())
    clf = xgb.XGBClassifier(max_depth=int(max_depth),
                            learning_rate=learning_rate,
                            n_estimators=int(n_estimators),
                            gamma=gamma,
                            reg_alpha=reg_alpha,
                            reg_lambda=reg_lambda)
    result = np.mean(cross_val_score(clf, X, y, cv=5, scoring='roc_auc', 
                                     verbose=0, n_jobs=4))
    return result

BO = BayesianOptimization(f=xgb_eval,
                          pbounds={'max_depth': (3,7),
                                   'learning_rate': (5e-2, 3e-1),
                                   'n_estimators': (50,300),
                                   'gamma': (0,5),
                                   'reg_alpha': (0,5),
                                   'reg_lambda': (1e-1, 5)})

# optimize
BO.maximize(init_points=3, n_iter=10)

history_df = pd.DataFrame.from_dict(BO.res)
history_df.to_csv('BO-AUC-5fold-run-01-v1.csv')