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

# from julia import OptimalTrees
# from julia import DataFrames

wordgroups = list(pd.read_excel('wordgroups.xlsx')['Word Groupings'])

#os.chdir(os.path.expanduser("~/Dropbox (Partners HealthCare)/MVP/Radiology NLP Project/Radiology Report Labeling/Stroke Labeling/Labeled Reports"))
labeled = pd.read_csv("Stroke Labeling/Labeled Reports/Batch 1/Batch1_Stroke_final.csv")
labeled = labeled[['EMPI','Report_Number','Report_Date_Time','Ischemic Stroke?','Repeat Reports','Report_Text']]
print(labeled.shape)

# Keep only those we have labeled 0 or 1 for Ischemic Stroke
labeled = labeled[(labeled['Ischemic Stroke?'] == 0) | (labeled['Ischemic Stroke?'] == 1)]
print(labeled.shape)

# Keep only those with IMPRESSIONS
labeled = labeled.iloc[[x for x in range(labeled.shape[0]) if 'IMPRESSION:' in labeled.Report_Text.iloc[x]]]
print(labeled.shape)

# replace whitespace with space
labeled['Report_Text'] = labeled['Report_Text'].apply(lambda text: ' '.join(text.split()))

# for x in labeled[labeled.EMPI == 105035731].Report_Text:
#     print(x)

# Remove header parts
# sum(['-'*78 in x or 'HISTORY:' in x or 'REPORT' in x for x in labeled.Report_Text])/len(labeled.Report_Text)
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text: re.split('-'*78, text, 1)[-1])
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text: re.split('HISTORY:', text, 1)[-1])
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text: re.split('REPORT ', text, 1)[-1])
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text: re.split('REPORT:', text, 1)[-1])

# Remove footer parts
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text:
                                                   re.split('electronically signed by:', text, flags=re.IGNORECASE)[0])
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text:
                                                   ''.join(re.split('i, the teaching physician, have reviewed the images and agree with the report as written', text, flags=re.IGNORECASE)))
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text:
                                                   re.split('radiologists: signatures:', text, flags=re.IGNORECASE)[0])
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text:
                                                   re.split('providers: signatures:', text, flags=re.IGNORECASE)[0])
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text:
                                                   re.split('findings were discussed on', text, flags=re.IGNORECASE)[0])
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text:
                                                   re.split('this report was electronically signed by', text, flags=re.IGNORECASE)[0])

# Remove reference texts =====
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text: ''.join([x for i,x in enumerate(text.split('='*34)) if i != 1]))


# Get just IMPRESSION, convert both to lower
labeled['impression'] = labeled.Report_Text.apply(lambda text: text.split('IMPRESSION:')[1])
labeled['impression'] = labeled.impression.apply(lambda text: text.lower())
labeled['Report_Text'] = labeled.Report_Text.apply(lambda text: text.lower())

# Replace ngrams in Report_Text & IMPRESSION with their units
for group in wordgroups:
    labeled['impression'] = labeled.impression.apply(lambda text: text.replace(group, ''.join(group.split())))
    labeled['Report_Text'] = labeled.Report_Text.apply(lambda text: text.replace(group, ''.join(group.split())))

# CHECK WITH CHARLENE THAT IMPRESSIONS SHOULD NOT BE IDENTICAL AND SAME DAY
# TODO: later pull out just the date and just the time field, check for dupes on those
labeled = labeled.drop_duplicates(subset=['EMPI','Report_Date_Time','impression'])
labeled = labeled.reset_index(drop=True)
print(labeled.shape)
# print(labeled[labeled.Report_Date_Time == '10/18/14 10:20'].Report_Text.iloc[0])
labeled1 = labeled

#os.chdir(os.path.expanduser("~/Dropbox (Partners HealthCare)/MVP/Radiology NLP Project/Radiology Report Labeling/Stroke Labeling/Labeled Reports/"))
labeled2 = pd.read_csv("Stroke Labeling/Labeled Reports/Batch 2/Batch2_Stroke_final.csv")
# labeled2 = labeled2[['EMPI','Report_Date_Time','Ischemic Stroke?','Repeat Reports','Report_Text']]
print(labeled2.shape)

# Keep only those we have labeled2 0 or 1 for Ischemic Stroke
labeled2 = labeled2[(labeled2['Ischemic Stroke?'] == 0) | (labeled2['Ischemic Stroke?'] == 1)]
print(labeled2.shape)

# Keep only those with IMPRESSIONS
labeled2 = labeled2.iloc[[x for x in range(labeled2.shape[0]) if 'IMPRESSION:' in labeled2.Report_Text.iloc[x]]]
print(labeled2.shape)

# replace whitespace with space
labeled2['Report_Text'] = labeled2['Report_Text'].apply(lambda text: ' '.join(text.split()))

# for x in labeled2[labeled2.EMPI == 105035731].Report_Text:
#     print(x)

# Remove header parts
# # sum(['-'*78 in x or 'HISTORY:' in x or 'REPORT' in x for x in labeled2.Report_Text])/len(labeled2.Report_Text)
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text: re.split('-'*78, text, 1)[-1])
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text: re.split('HISTORY:', text, 1)[-1])
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text: re.split('REPORT ', text, 1)[-1])
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text: re.split('REPORT:', text, 1)[-1])

# Remove footer parts
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text:
#                                                    re.split('electronically signed by:', text, flags=re.IGNORECASE)[0])
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text:
#                                                    ''.join(re.split('i, the teaching physician, have reviewed the images and agree with the report as written', text, flags=re.IGNORECASE)))
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text:
#                                                    re.split('radiologists: signatures:', text, flags=re.IGNORECASE)[0])
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text:
#                                                    re.split('providers: signatures:', text, flags=re.IGNORECASE)[0])
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text:
#                                                    re.split('findings were discussed on', text, flags=re.IGNORECASE)[0])
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text:
#                                                    re.split('this report was electronically signed by', text, flags=re.IGNORECASE)[0])

# # Remove reference texts =====
# labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text: ''.join([x for i,x in enumerate(text.split('='*34)) if i != 1]))


# Get just IMPRESSION, convert both to lower
labeled2['impression'] = labeled2.Report_Text.apply(lambda text: text.split('IMPRESSION:')[1])
labeled2['impression'] = labeled2.impression.apply(lambda text: text.lower())
labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text: text.lower())

# Replace ngrams in Report_Text & IMPRESSION with their units
for group in wordgroups:
    labeled2['impression'] = labeled2.impression.apply(lambda text: text.replace(group, ''.join(group.split())))
    labeled2['Report_Text'] = labeled2.Report_Text.apply(lambda text: text.replace(group, ''.join(group.split())))

# CHECK WITH CHARLENE THAT IMPRESSIONS SHOULD NOT BE IDENTICAL AND SAME DAY
# TODO: later pull out just the date and just the time field, check for dupes on those
# labeled2 = labeled2.drop_duplicates(subset=['EMPI','Report_Date_Time','impression'])
labeled2 = labeled2.reset_index(drop=True)
print(labeled2.shape)

labeled = pd.concat([labeled1[['Report_Number', 'Report_Text','impression','Ischemic Stroke?']],
           labeled2[['Report_Number', 'Report_Text','impression','Ischemic Stroke?']]], ignore_index=True)
print("Total:", labeled.shape)

#os.chdir(os.path.expanduser("~/Dropbox (Partners HealthCare)/MVP/RPDR/Ischemic_stroke_2003-2018/Ischemic Stroke 2014-2018/Unstructured"))
#raw = pd.read_csv("Rad_processed.csv", header=0)
#ct_reports = raw[raw.Report_Description.isin(ct_types)].Report_Number
ct_reports = pd.read_csv("ct_report_numbers.csv", header=None)
#mr_reports = raw[raw.Report_Description.isin(mr_types)].Report_Number
mr_reports = pd.read_csv("mr_report_numbers.csv", header=None)
labeled_ct = labeled[labeled.Report_Number.isin(ct_reports[0])]
labeled_mr = labeled[labeled.Report_Number.isin(mr_reports[0])]
print("CT:", labeled_ct.shape)
print("MR:", labeled_mr.shape)

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
    tuned_parameters = [{'max_depth':range(3,maxDepth+1), 'min_samples_leaf':range(1,minBucket+1)}]
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
    print (clf.best_params_)


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

def featurize(labeled, technique='bow', impression_only=True,
              d=100, window=10, corpus='Yousem_StrokeXs_UpToDate_Rad2010', sentence='TRUE'):
    if technique == 'glove':
        #os.chdir(os.path.expanduser("~/Dropbox (Partners HealthCare)/MVP/RPDR/GloVe/Training Radiology and Stroke Resources/Vector_Representations"))
        dataset = corpus + '_' + str(d) + 'dim_' + str(50) + 'iter_' + str(window) + 'window_5min_' + sentence + 'sent'
        dataset = "GloVe/Training Radiology and Stroke Resources/Vector_Representations/" + dataset
        raw = pd.read_csv(dataset + '.csv', header=0)
        df = raw.set_index(raw.columns[0])
        df.index = [str(x) for x in df.index]
        df = df.reindex(sorted(df.index))

        # Create new dataframe with just vectors as features
        columns = ['V'+str(i+1) for i in range(d)]
        X = pd.DataFrame(index=labeled.index , columns=columns)

        # For each report, take the avg of the words in 'IMPRESSION'
        for i in range(labeled.shape[0]):
            avg = np.array([0]*d)
            count = 0
            if impression_only:
                text = labeled.impression.iloc[i]
            else:
                text = labeled.Report_Text.iloc[i]
            text = "".join([char for char in text if char not in string.punctuation])
            text = text.split()
            # Take the avg
            for word in text:
                if word in df.index:
                    vec = np.array(df.loc[word])
                    avg = avg + vec
                    count += 1
            avg = avg/count
            X.iloc[i] = avg

    elif technique == 'bow':
        vectorizer = CountVectorizer()
        if impression_only:
            features = vectorizer.fit_transform(labeled.impression).todense()
        else:
            features = vectorizer.fit_transform(labeled.Report_Text).todense()
        vocab_counts = vectorizer.vocabulary_
        feature_names = vectorizer.get_feature_names()
        #Each one of the elements of the table
        X = pd.DataFrame(features, columns = feature_names)
        #Now every row corresponds to a document and we have a vector of words that counts just the frequency in the matrix.

    elif technique == 'tfidf':
        vectorizer = CountVectorizer()
        if impression_only:
            features = vectorizer.fit_transform(labeled.impression).todense()
        else:
            features = vectorizer.fit_transform(labeled.Report_Text).todense()
        vocab_counts = vectorizer.vocabulary_
        feature_names = vectorizer.get_feature_names()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(features).toarray()
        X = pd.DataFrame(tfidf, columns = feature_names)

    else:
        raise ValueError('Technique specified not valid')

    return X

"""
Run Classification
"""

y = labeled_ct['Ischemic Stroke?']
num_seeds = 5

#os.chdir(os.path.dirname(__file__))
ouput_file = open("results.txt", "w")
for technique in ['bow','tfidf','glove']:
    for classifier in ['knn','CART', 'RF', 'logreg']:
#    for classifier in ['knn', 'logreg']:
        print('='*10, 'Results for', technique, classifier, '='*10)
        auc = 0
        for seed in range(1, num_seeds+1):
            print('Results for seed',seed)
            X = featurize(labeled_ct, technique, impression_only=False, d=200)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

            verbose = False

            if classifier == 'knn':
                k_list = [5, 10, 15, 20]
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
                tolerance = 0.01
                score = 'roc_auc'
                y_proba = lasso(tolerance, verbose, X_test, y_test, X_train, y_train)

            y_pred = (y_proba > 0.5).astype(int)
            print(roc_auc_score(y_test, y_proba))
            ouput_file.write(technique + ',' + classifier + ',' + str(seed) + ',' + str(roc_auc_score(y_test, y_proba)) + '\n')
            auc = auc + roc_auc_score(y_test, y_proba)
        print(technique, classifier, 'Average AUC:', auc/num_seeds)
        # ouput_file.write(technique + ',' + classifier + ',' + str(seed) + ',' + str(roc_auc_score(y_test, y_proba)))
ouput_file.close()
