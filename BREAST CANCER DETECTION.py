#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[72]:


df=pd.read_csv('data.csv')


# In[73]:


df.head()


# In[74]:


# M:-harmfull
# b:-Non-harmfull


# In[75]:


df.tail()


# In[76]:


df.isnull().sum()


# In[77]:


del df['Unnamed: 32']


# In[78]:


df.dtypes.count()


# In[79]:


df.shape


# In[80]:


df['diagnosis'].value_counts()


# In[81]:


sns.countplot(df['diagnosis'], label="count")


# In[82]:


df.dtypes


# In[83]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
df.iloc[:,1]=labelencoder_y.fit_transform(df.iloc[:,1].values)


# In[84]:


sns.pairplot(df.iloc[:,1:6])


# In[85]:


sns.pairplot(df.iloc[:,1:6], hue = 'diagnosis')


# In[86]:


df.head()


# In[87]:


df.iloc[:,1:32].corr()


# In[88]:


sns.heatmap(df.iloc[:,1:31].corr())


# In[89]:


plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt='0.0%')


# In[90]:


# X-> independent variable (features that can detect the cancer)
# Y-> dependent variable (target value) (diagnosis)
X=df.iloc[:,2:31].values
Y=df.iloc[:,1].values


# In[91]:


# 75% train and 25% test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test=train_test_split(X,Y, test_size=0.25 , random_state=0)


# In[92]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[104]:


def models(X_train, Y_train):
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion = 'entropy' , random_state=0)
    tree.fit(X_train,Y_train)
    
    from sklearn.ensemble import RandomForestClassifier
    forest= RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    
    print(log.score(X_train ,Y_train))
    print(tree.score(X_train ,Y_train))
    print(forest.score(X_train ,Y_train))
    
    return log, tree ,forest


# In[106]:


# Trainig model score
model=models(X_train ,Y_train)


# In[136]:


#CONFUSION METRIX:-
# TP-84
# TN-50
# FP-4
# # FN-3
#Decision tree model have best accuracy in Train data
# Random forest model have best accuracy in Test data


# In[127]:


# Test model accuracy on confusion matrix

from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    print('Model',i)
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    TP=cm[0][0]
    TN=cm[1][1]
    FN=cm[1][0]
    FP=cm[0][1]

    print (cm)
    print("testing accuracy = ", (TP + TN)/(TP + TN + FP + FN))
    print()


# In[137]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report(Y_test,model[0].predict(X_test)))
print(accuracy_score(Y_test,model[0].predict(X_test)))
print()
print()
print()
print(classification_report(Y_test,model[1].predict(X_test)))
print(accuracy_score(Y_test,model[1].predict(X_test)))
print()
print()
print()
print(classification_report(Y_test,model[2].predict(X_test)))
print(accuracy_score(Y_test,model[2].predict(X_test)))


# In[142]:


# print the prediction of random forest classifier model
pred = model[2].predict(X_test)
print(pred)
print()
print(Y_test)

