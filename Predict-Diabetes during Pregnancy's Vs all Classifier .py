#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install sklearn')


# In[2]:


import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[3]:


dataset=pd.read_csv("diabetes2.csv", delimiter=",")


# In[4]:


dataset.describe()


# In[5]:


X = dataset[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = dataset[['Outcome']]


# In[6]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=0)


# # Decision Tree Classifier Model

# In[7]:


model=DecisionTreeClassifier(random_state=84,splitter='best', max_features=8)

print("[INFO] training model...")
model.fit(X_train,y_train)

print("[INFO] evaluating...")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# # Random Forest Classifier Model

# In[8]:


model=RandomForestClassifier(n_estimators=10, random_state=42, max_features="auto")

print("[INFO] training model....")
model.fit(X_train,y_train)

print("[INFO] evaluating...")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# # KNeighbors Classifier Model

# In[9]:


model=KNeighborsClassifier(n_neighbors=9)

print("[INFO] training model....")
model.fit(X_train,y_train)

print("[INFO] evaluating...")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# # SVC Classifier Model

# In[10]:


model=SVC(kernel="linear",C=1)

print("[INFO] training model....")
model.fit(X_train,y_train)

print("[INFO] evaluating...")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# # Logistic Regression Classifier Model

# In[11]:


model=LogisticRegression(max_iter=100,solver='liblinear')

print("[INFO] training model....")
model.fit(X_train,y_train)

print("[INFO] evaluating...")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# # Logistic Regression Classifier Model gives a better Accuracy % compared to all other Classifier Models

# # Assumption - As we increase the size of Test sample Logistic Regression Model becomes more reliable classifier model to predict target variable with high Accuracy percentage

# In[ ]:





# In[ ]:




