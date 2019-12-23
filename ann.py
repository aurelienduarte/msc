#!/usr/bin/env python
from keras import models
from keras import layers
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pickle

df2 = pd.read_csv('data/ip2.nmap.online.hosts.masscan.csv.optimised.som_win_map8.csv')
df2 = pd.read_csv('data/ip2.nmap.online.hosts.masscan.csv.optimised.som_win_map6.csv')

df2_data=df2.drop(columns=['cluster'])
df2_labels=df2['cluster'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df2_data, to_categorical(df2_labels,num_classes=64), test_size=0.3)

classifier = Sequential()
#Hidden Layer
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal', input_dim=X_train.shape[1]))
#Output Layer
classifier.add(Dense(64, activation='softmax', kernel_initializer='random_normal'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='categorical_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=50, epochs=50, use_multiprocessing=True)
eval_model=classifier.evaluate(X_train, y_train, use_multiprocessing=True)
eval_model

y_pred=classifier.predict(X_test)
# y_pred =(y_pred>0.5)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)

print('accuracy_score',accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
print('recall_score',recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1),average='weighted'))
print('f1_score',f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1),average='weighted'))
print('precision_score',precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1),average='weighted'))

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.classifier.16.map6', 'wb') as outfile:
    pickle.dump(classifier, outfile)

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.test.16.map6', 'wb') as outfile:
    pickle.dump(X_test, outfile)

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.train.16.map6', 'wb') as outfile:
    pickle.dump(X_train, outfile)

with open('data/ip2.nmap.online.hosts.masscan.csv.optimised.clusters.16.map6', 'wb') as outfile:
    pickle.dump(y_train, outfile)
