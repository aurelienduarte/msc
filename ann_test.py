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

nrow=X_test.iloc[[110]]
for i in nrow:
    print(nrow[i])

y_pred=classifier.predict(nrow)
cluster_pred=y_pred.argmax(axis=1)
y_test.argmax(axis=1)
df2.loc[5075]

#get all open ports in the sample
nrow.loc[:, nrow.eq(1).all()]

nrow0nan['port5985']=0
nrow0nan['port47001']=0
nrow0nan['port49153']=0
nrow0nan['port49152']=0
nrow0nan['port80']=1
nrow0nan['port22']=1
nrow0nan.loc[:, nrow0nan.eq(1).all()]

nrow0nan_pred=classifier.predict(nrow0nan)
cluster_pred=nrow0nan_pred.argmax(axis=1)
cluster_pred
