#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from sklearn.naive_bayes import GaussianNB
from  email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
t0 = time()
gnb = GaussianNB()
y_fit = gnb.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
y_pred = y_fit.predict(features_test)
print "Predict time:", round(time()-t0, 3), "s"
#########################################################
from sklearn.metrics import accuracy_score

rst = accuracy_score(y_pred, labels_test)
print(rst)

