#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain')
df['Sex_Code'] = df['Sex'].map({'female':1, 'male':0}).astype('int')
df['Sex'] = df['Sex_Code']
df['Age'] = df['Age'].fillna(df['Age'].mean())
df


# In[2]:


# set infinite number of rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
df


# In[3]:


print(list(df["Survived"]))


# In[4]:


X = df[["Pclass", "Sex", "Age"]]
X


# In[5]:


from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2, random_state=2024)
clf.fit(X)


# In[6]:


y_pred = clf.predict(X)
print(y_pred)


# In[7]:


Jack = clf.predict([[3, 0, 23.0]])
print("Jack", Jack)
Rose = clf.predict([[1, 1, 20.0]])
print("Rose", Rose)


# In[8]:


two = {"Pclass":[3,1], "Sex":[0,1], "Age":[23.0,20.0]}
df1 = pd.DataFrame(two)
X_new = pd.concat([X, df1]).reset_index()
X_new.pop("index")
X_new


# In[9]:


from sklearn.cluster import DBSCAN
clf = DBSCAN(eps=1, min_samples=3)
clf.fit(X_new)


# In[10]:


print(clf.labels_)


# In[11]:


from sklearn.cluster import DBSCAN
clf = DBSCAN(eps=2, min_samples=3)
clf.fit(X_new)


# In[12]:


print(clf.labels_)


# In[13]:


from sklearn.cluster import AgglomerativeClustering
clf = AgglomerativeClustering(n_clusters=2, linkage='ward')
clf.fit(X_new)


# In[14]:


y_pred = clf.fit_predict(X_new)
print(y_pred)


# In[ ]:


from sklearn.cluster import BisectingKMeans
clf = BisectingKMeans(n_clusters=2, linkage='ward')
clf.fit(X_new)


# In[ ]:


y_pred = clf.fit_predict(X_new)
print(y_pred)

