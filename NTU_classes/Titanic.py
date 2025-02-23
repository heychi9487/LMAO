#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('http://bit.ly/kaggletrain')
df.head(10)


# In[2]:


df['Sex_Code'] = df['Sex'].map({'female':1, 'male':0}).astype('int')
df['Sex'] = df['Sex_Code']
df['Age'] = df['Age'].fillna(df['Age'].mean())
df


# In[3]:


X = df[["Pclass", "Sex", "Age"]]
y = df["Survived"]


# In[4]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 23.0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 20.0]])
print("Rose", Rose)


# In[5]:


# K-NN
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 23.0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 20.0]])
print("Rose", Rose)


# In[6]:


# SVC
from sklearn import svm
clf = svm.SVC()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 23.0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 20.0]])
print("Rose", Rose)


# In[7]:


# Gaussian Naive bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 23.0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 20.0]])
print("Rose", Rose)


# In[8]:


# Multinomail Naive bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 23.0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 20.0]])
print("Rose", Rose)


# In[9]:


# Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 23.0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 20.0]])
print("Rose", Rose)


# In[10]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,y)
Jack = clf.predict([[3, 0, 23.0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 20.0]])
print("Rose", Rose)


# In[11]:


# XGBoost
from xgboost.sklearn import XGBClassifier
clf = XGBClassifier()
clf.fit(X,y)
two = {"Pclass":[3,1], "Sex":[0,1], "Age":[23.0,20.0]}
df1 = pd.DataFrame(two)
#Jack = clf.predict([[3, 0, 23.0]])
Jack = clf.predict(df1.iloc[0:1])
print("Jack", Jack)
Rose = clf.predict(df1.iloc[1:2])
print("Rose", Rose)


# ## Practice
# If we respectively use max age and min age to replace empty age, will the results of Jack and Rose change?

# In[ ]:




