import os

import pandas as pd, numpy as np
import random
import csv, xlrd
import matplotlib.pyplot as plt
# %matplotlib inline
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('punkt')
import string
import operator
#import julia
import heapq
#import graphviz
from collections import Counter
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import tree
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

"""
Define classifiers below
"""
def cos_knn_cv(train_X, train_y, test_X, test_y, k_list, nfolds, verbose):

    # Encapsulated helper fcn
    def cos_knn(k, train_X, train_y, test_X, test_y):
        # find cosine similarity for every point in test_data between every other point in training
        cosim = cosine_similarity(test_X, train_X)

        # get top k indices of images in stored_data that are most similar to any given test_data point
        # note this returns locations, not pandas df indices
        top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]

        # convert indices to numbers using stored target values
    #     top = [[train_y[j] for j in i[:k]] for i in top]
        top = [[train_y.iloc[j] for j in i[:k]] for i in top]

        # vote and get prediction for every point in test_data
        pred = [max(set(i), key=i.count) for i in top]
        # vote, and return probability of positive class for every point in test_data
        proba = [sum(np.array(i) == 1) / len(i) for i in top]
        proba = np.array(proba)

        return proba


    kf = KFold(n_splits=nfolds, random_state=19)
    cv_scores = [0]*len(k_list)
    for train_index, test_index in kf.split(train_X):
        X_train_cv, y_train_cv = train_X.iloc[train_index], train_y.iloc[train_index]
        X_test_cv, y_test_cv = train_X.iloc[test_index], train_y.iloc[test_index]
        for i in range(len(k_list)):
            y_proba_cv = cos_knn(k_list[i], X_train_cv, y_train_cv, X_test_cv, y_test_cv)
            y_pred_cv = (y_proba_cv > 0.5).astype(int)
#             cv_scores[i] += f1_score(y_test_cv, y_pred_cv)
            cv_scores[i] += roc_auc_score(y_test_cv, y_proba_cv)
    # Find best k
    best_k = k_list[cv_scores.index(max(cv_scores))]
    print('Best # of neighbors:', best_k)

    # Call helper on all data using best k
    proba = cos_knn(best_k, train_X, train_y, test_X, test_y)
    pred = (proba > 0.5).astype(int)

    # print table giving classifier accuracy using test_target
    if verbose:
        print(classification_report(test_y, pred))

    return proba

def greedy_tree(maxDepth, minBucket, score, verbose, test_X, test_y, train_X, train_y):
    tuned_parameters = [{'max_depth':range(3,maxDepth+1), 'min_samples_leaf':range(2,minBucket+1)}]
    clf = GridSearchCV(tree.DecisionTreeClassifier(criterion="gini", random_state=100, splitter = 'best'),
                           tuned_parameters, cv=10,  iid=True, scoring = score)
    clf.fit(train_X, train_y)
    print (clf.best_params_)

    pred = clf.predict(test_X)
    proba = clf.predict_proba(test_X)[:,1]
    if verbose:
        print(classification_report(test_y, pred))
    return proba

def forest(maxTrees, maxDepth, minBucket, score, verbose, test_X, test_y, train_X, train_y):
    tuned_parameters = [{'n_estimators': range(10,maxTrees+10,100),'max_depth':range(3,maxDepth+1), 'min_samples_leaf':range(1,minBucket+1), 'bootstrap' : [True, False]}]
    clf = GridSearchCV(RandomForestClassifier(criterion="gini", random_state=100),
                           tuned_parameters, cv=10,  iid=True, scoring = score)
    clf.fit(train_X, train_y)
    print (clf.best_params_)

    pred = clf.predict(test_X)
    proba = clf.predict_proba(test_X)[:,1]
    if verbose:
        print(classification_report(test_y, pred))
    return proba

def lasso(tolerance, verbose, test_X, test_y, train_X, train_y):
    tuned_parameters =[{'tol': np.arange(0.0001, tolerance + 0.01, 0.01)}]
    clf = GridSearchCV(LogisticRegression(penalty = 'l1',max_iter=1000, random_state=100), tuned_parameters, cv = 10, iid=True, scoring = score)
    clf.fit(train_X, train_y)
    print(clf.best_params_)


    pred = clf.predict(test_X)
    proba = clf.predict_proba(test_X)[:,1]
    if verbose:
        print(classification_report(test_y, pred))
        features= abs(clf.best_estimator_.coef_[0]) > 0.3
        coef = pd.DataFrame()
        coef['word'] = test_X.columns[features]
        coef['coef'] = clf.best_estimator_.coef_[0,features]
        #df_c = pd.concat([Test.coef_[0,features].reset_index(drop=True), X_test.columns[features]], axis=1)
        print(coef.sort_values('coef', ascending= False))
    return proba

"""
Run Classification
"""

num_seeds = 5
#os.chdir(os.path.dirname(__file__))
ouput_file = open("classifier_location_results.txt", "w")
for technique in ['bow','tfidf','glove']:
    for classifier in ['knn','CART', 'RF', 'logreg']:
#    for classifier in ['knn', 'logreg']:
        print('='*10, 'Results for', technique, classifier, '='*10)
        auc = 0
        for seed in range(1, num_seeds+1):
            print('Results for seed',seed)
            # X = featurize(labeled_ct, technique, impression_only=False, d=200)
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
            X_train = pd.read_csv("clean_data/location/X_train_" + technique + '_' + str(seed) + '.csv')
            X_test = pd.read_csv("clean_data/location/X_test_" + technique + '_' + str(seed) + '.csv')
            y_train = pd.read_csv("clean_data/location/y_train_" + technique + '_' + str(seed) + '.csv', header=None)
            y_test = pd.read_csv("clean_data/location/y_test_" + technique + '_' + str(seed) + '.csv', header=None)
            y_train = y_train.iloc[:,0]
            y_test = y_test.iloc[:,0]

            verbose = False

            if classifier == 'knn':
                k_list = [5, 10, 15, 20, 25]
                nfolds = 5
                y_proba = cos_knn_cv(X_train, y_train, X_test, y_test, k_list, nfolds, verbose)

            if classifier == 'CART':
                maxDepth = 10
                minBucket = 10
                score = 'roc_auc'
                y_proba = greedy_tree(maxDepth, minBucket, score, verbose, X_test, y_test, X_train, y_train)

            if classifier == 'RF':
                maxTrees = 200
                maxDepth = 10
                minBucket = 10
                score = 'roc_auc'
                y_proba = forest(maxTrees, maxDepth, minBucket, score, verbose, X_test, y_test, X_train, y_train)

            if classifier == 'logreg':
                tolerance = 0.1
                score = 'roc_auc'
                y_proba = lasso(tolerance, verbose, X_test, y_test, X_train, y_train)

            y_pred = (y_proba > 0.5).astype(int)
            np.savetxt("proba_location/proba_" + technique + "_" + classifier + "_" + str(seed) + ".csv", y_proba, delimiter=",")
            print(roc_auc_score(y_test, y_proba))
            ouput_file.write(technique + ',' + classifier + ',' + str(seed) + ',' + str(roc_auc_score(y_test, y_proba)) + '\n')
            auc = auc + roc_auc_score(y_test, y_proba)
        print(technique, classifier, 'Average AUC:', auc/num_seeds)
        # ouput_file.write(technique + ',' + classifier + ',' + str(seed) + ',' + str(roc_auc_score(y_test, y_proba)))
ouput_file.close()
