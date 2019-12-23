#!/usr/bin/env python
from keras import models
from keras import layers
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

df2 = pd.read_csv('data/ip2.nmap.online.hosts.masscan.csv.optimised.som_win_map6.csv')

df2_data=df2.drop(columns=['cluster'])
df2_labels=df2['cluster'].values

results=dict([
    (8,  {'accuracy':[],'recall':[],'f1':[],'precision':[]}), 
    (12, {'accuracy':[],'recall':[],'f1':[],'precision':[]}), 
    (16, {'accuracy':[],'recall':[],'f1':[],'precision':[]}), 
    (32, {'accuracy':[],'recall':[],'f1':[],'precision':[]}), 
    (64, {'accuracy':[],'recall':[],'f1':[],'precision':[]}), 
    ])

for units in results.keys():
    for run in range(1,11):
        X_train, X_test, y_train, y_test = train_test_split(df2_data, to_categorical(df2_labels,num_classes=64), test_size=0.3)
        #
        classifier = Sequential()
        classifier.add(Dense(units, activation='relu', kernel_initializer='random_normal', input_dim=X_train.shape[1]))
        classifier.add(Dense(64, activation='softmax', kernel_initializer='random_normal'))
        classifier.compile(optimizer ='adam',loss='categorical_crossentropy', metrics =['accuracy'])
        #
        #Fitting the data to the training dataset
        classifier.fit(X_train,y_train, batch_size=50, epochs=50,use_multiprocessing=True)
        eval_model=classifier.evaluate(X_train, y_train,use_multiprocessing=True)
        eval_model
        y_pred=classifier.predict(X_test)
        # y_pred =(y_pred>0.5)
        cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        # print(cm)
        # np.sum(cm.diagonal())
        # np.sum(cm)
        #
        print('Units:',units,", Run:",run)
        results[units]['accuracy'].append(accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
        results[units]['recall'].append(recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1),average='weighted'))
        results[units]['f1'].append(f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1),average='weighted'))
        results[units]['precision'].append(precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1),average='weighted'))

for units in results.keys():
    accuracy=np.mean(results[units]['accuracy'])
    recall=np.mean(results[units]['recall'])
    f1=np.mean(results[units]['f1'])
    precision=np.mean(results[units]['precision'])
    print(units,accuracy,recall,f1,precision)
