import os
# Change this path depending on where you're running
path = "/nfs/sloanlab001/projects/edema-partners_proj/LSTM/src"
os.chdir(path)
print(os.getcwd())
import pandas as pd, numpy as np
import random
import csv, xlrd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
import string
import operator
import heapq
from collections import Counter
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from keras import models
from keras import layers
import seaborn as sns
import functools
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM
import keras.backend as K
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

class myarray(np.ndarray):
	def __new__(cls, *args, **kwargs):
		return np.array(*args, **kwargs).view(myarray)
	def index(self, value):
		return np.where(self == value)

def colselect_X(df, q):
	#We need to find for each row how many columns are non zero.
	l = []
	for i in range(0, (df.shape[0]-1)):
		a = myarray(df[i])
		x = a.index(0)
		if len(x[0])>1 :
			l.append(x[0][0])

	r = np.asarray(l)
	return np.quantile(r, q)


def runSingle(batch_size, num_epoch, lstm_size, activation_func, dropout_u, dropout_w,seed):

	#Split between training and validation set
	X_big = pd.read_csv('../../clean_data/stroke/X_train_bow_'+ str(seed) + '.csv', header=1)
	X_test = pd.read_csv('../../clean_data/stroke/X_test_bow_'+ str(seed) + '.csv', header=1)
	y_big = pd.read_csv('../../clean_data/stroke/y_train_bow_'+ str(seed) + '.csv', header=0)
	y_test = pd.read_csv('../../clean_data/stroke/y_test_bow_'+ str(seed) + '.csv', header=0)
	X_train, X_val, y_train, y_val = train_test_split(X_big, y_big, test_size=0.33, random_state=19)

	quant = 0.9
	df = pd.DataFrame(columns=['quant', 'lstm_size', 'dropout','recurrent_dropout','batch_size',
							   'num_epoch','activation_func','test_loss','test_accuracy',
							   'test_auc'])
	dataset_name = 'BOW_Results_quant_'+str(quant)+'_lstm_size_'+str(lstm_size)+'_dropout_w_'+str(dropout_w)+'_dropout_u_'+str(dropout_u)+'_batch_size_'+str(batch_size)+'_num_epoch_'+str(num_epoch)+'_activation_func_'+str(activation_func)+'_seed_'+str(seed)+'.csv'
	dataset_proba_name = 'BOW_proba_quant_'+str(quant)+'_lstm_size_'+str(lstm_size)+'_dropout_w_'+str(dropout_w)+'_dropout_u_'+str(dropout_u)+'_batch_size_'+str(batch_size)+'_num_epoch_'+str(num_epoch)+'_activation_func_'+str(activation_func)+'_seed_'+str(seed)+'.csv'

	#Parameter Selection
	print('Build model...')
	model = Sequential()
	# e = Embedding(vocab_size, d, input_length=X_big.shape[1])
	vocab_size = X_big.shape[1]
	nb_classes = 1
	e = Embedding(vocab_size, lstm_size, input_length=X_big.shape[1])
	model.add(e)
	model.add(LSTM(lstm_size, dropout=dropout_w, recurrent_dropout=dropout_u))
	model.add(Dense(nb_classes, activation=activation_func))

	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	print('Train...')
	model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch,validation_data=(X_val, y_val))
	score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)
	print("Generating test predictions...")
	preds = model.predict_classes(X_test, verbose=0)
	#Get the auc
	y_pred_keras = model.predict(X_test).ravel()
	np.savetxt('../results/'+dataset_proba_name, y_pred_keras, delimiter=",")
	fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

	auc_keras = auc(fpr_keras, tpr_keras)
	print('AUC of the model', auc_keras)

	df.loc[0] = [quant, lstm_size, dropout_w,dropout_u,batch_size,
				 num_epoch,activation_func,score,acc,auc_keras]

	print(dataset_name)
	df.to_csv('../results/'+dataset_name)
	# preds.to_csv('../results/'+dataset_probs_name)
	print(df)
