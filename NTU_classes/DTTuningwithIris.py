#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # randomly split data to training and testing data

from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target


# In[2]:


# Decision Tree (Split data randomly from 90~10%)
ratio = 100
ratiovalues = [i for i in range(10, ratio, 10)]
train_scores = []
test_scores = []

for i in ratiovalues:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i/100, random_state=71)

    clf = tree.DecisionTreeClassifier(random_state = 71)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train) #train
    train_acc = accuracy_score(y_pred_train, y_train)
    train_scores.append(train_acc)
    
    y_pred_test = clf.predict(X_test) #test
    test_acc = accuracy_score(y_pred_test, y_test)
    test_scores.append(test_acc)
    
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

plt.plot(ratiovalues, train_scores, '-o', label='Train')
plt.plot(ratiovalues, test_scores, '-o', label='Test')
plt.legend()
plt.show()


# In[3]:


# Decision Tree (Split data randomly by 90% vs. 10% + different depth)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=71) #train_size = 0.8

depth = 6
depthvalues = [i for i in range(1, depth)]
train_scores = []
test_scores = []

for i in depthvalues:
    clf = tree.DecisionTreeClassifier(random_state = 71, max_depth = i)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train) #train
    train_acc = accuracy_score(y_pred_train, y_train)
    train_scores.append(train_acc)
    
    y_pred_test = clf.predict(X_test) #test
    test_acc = accuracy_score(y_pred_test, y_test)
    test_scores.append(test_acc)
    
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

plt.plot(depthvalues, train_scores, '-o', label='Train')
plt.plot(depthvalues, test_scores, '-o', label='Test')
plt.legend()
plt.show()


# In[4]:


# Decision Tree (Split data randomly by 90% vs. 10% + different min sample leaf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=71) #train_size = 0.8

leaf = 10
leafvalues = [i for i in range(1, leaf)]
train_scores = []
test_scores = []

for i in leafvalues:
    clf = tree.DecisionTreeClassifier(random_state = 71, min_samples_leaf = i)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train) #train
    train_acc = accuracy_score(y_pred_train, y_train)
    train_scores.append(train_acc)
    
    y_pred_test = clf.predict(X_test) #test
    test_acc = accuracy_score(y_pred_test, y_test)
    test_scores.append(test_acc)
    
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

plt.plot(leafvalues, train_scores, '-o', label='Train')
plt.plot(leafvalues, test_scores, '-o', label='Test')
plt.legend()
plt.show()


# In[5]:


# Decision Tree (Split data randomly by 90% vs. 10% + different depth + different min sample leaf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=71) #train_size = 0.8

depth = 6
depthvalues = [i for i in range(1, depth)]
leaf = 10
leafvalues = [i for i in range(1, leaf)]
relative_best_train_score = 0
relative_best_test_score = 0
relative_best_depth = 0
relative_best_leaf = 0

for i in depthvalues:
    for j in leafvalues:
        clf = tree.DecisionTreeClassifier(random_state = 71, max_depth = i, min_samples_leaf = j)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train) #train
        train_acc = accuracy_score(y_pred_train, y_train)
        y_pred_test = clf.predict(X_test) #train
        test_acc = accuracy_score(y_pred_test, y_test)

        if ((train_acc > relative_best_train_score) and (test_acc > relative_best_test_score)):
            relative_best_train_score = train_acc
            relative_best_test_score = test_acc
            relative_best_depth = i
            relative_best_leaf = j

print("best depth:", relative_best_depth, "best min_sample_leaf:", relative_best_leaf,       "\nTraining score:", relative_best_train_score, "Testing score:", relative_best_test_score)


# In[6]:


# Decision Tree (Split data randomly from 90~10% + different depth + different min sample leaf)
ratio = 100
ratiovalues = [i for i in range(10, ratio, 10)]
depth = 6
depthvalues = [i for i in range(1, depth)]
leaf = 10
leafvalues = [i for i in range(1, leaf)]
relative_best_train_score = 0
relative_best_test_score = 0
relative_best_ratio = 0
relative_best_depth = 0
relative_best_leaf = 0

for k in ratiovalues:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = k/100, random_state=71)
    for i in depthvalues:
        for j in leafvalues:
            clf = tree.DecisionTreeClassifier(random_state = 71, max_depth = i, min_samples_leaf = j)
            clf.fit(X_train, y_train)
            y_pred_train = clf.predict(X_train) #train
            train_acc = accuracy_score(y_pred_train, y_train)
            y_pred_test = clf.predict(X_test) #train
            test_acc = accuracy_score(y_pred_test, y_test)

            if ((train_acc > relative_best_train_score) and (test_acc > relative_best_test_score)):
                relative_best_train_score = train_acc
                relative_best_test_score = test_acc
                relative_best_ratio = k
                relative_best_depth = i
                relative_best_leaf = j

print("best ratio of testing data:", relative_best_ratio, "best depth:", relative_best_depth, "best min_sample_leaf:", relative_best_leaf,       "\nTraining score:", relative_best_train_score, "Testing score:", relative_best_test_score)


# # Replace Iris dataset to Cancer dataset for practice
