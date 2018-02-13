# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:54:57 2018

@author: dell
"""

import os
import numpy as np
import pandas as pd
from sklearn import linear_model 
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib
os.chdir("f:/datafiles")  
train=pd.read_csv("mnist_train.csv",header=None)
test=pd.read_csv("mnist_test.csv",header=None)

#to chck how our data is
xtrain=train.iloc[:,1:]        #0th column is label which contains the no. for which this row data we have, so removed
ytrain=train.iloc[:,0]         #0th of ytrain will tell us what no. is this actually
xtest=test.iloc[:,1:] 
ytest=test.iloc[:,0]
ytrain5=(ytrain==5).astype(np.int)
ytest5=(ytest==5).astype(np.int)

arow=xtrain.iloc[0,:].reshape(28,28)       #this will signify one digit in whole, taking one complete row at a time
plt.imshow(arow,cmap=matplotlib.cm.binary,interpolation='nearest')
ytrain[0]

sgd=linear_model.SGDClassifier(random_state=42)
sgd.fit(xtrain,ytrain5)
ytrain5.value_counts()
predicted=sgd.predict(xtrain)

sgd1=linear_model.SGDClassifier(random_state=42)
sgd1.fit(xtest,ytest5)
ytest5.value_counts()
predicted1=sgd1.predict(xtest)

conf_mat=metrics.confusion_matrix(ytest5,predicted1)
print(conf_mat)
yscore=sgd.decision_function(xtest)
predicted2=(yscore>=0).astype(np.int) #by putting constraint on y-score we can manage threshold limit
print(predicted2)
conf_mat1=metrics.confusion_matrix(ytest5,predicted1)
print(conf_mat1)


#accuracy=96.3%   precission=75.1%  


