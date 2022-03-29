#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv('data.csv')


# In[8]:


df.head()


# In[15]:


# M:-harmfull
# b:-Non-harmfull


# In[16]:


df.tail()


# In[26]:


df.isnull().sum()


# In[27]:


del df['Unnamed: 32']


# In[39]:


df.dtypes.count()


# In[40]:


df.shape


# In[49]:


df['diagnosis'].value_counts()


# In[51]:


sns.countplot(df['diagnosis'], label="count")


# In[53]:


df.dtypes


# In[64]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
df.iloc[:,1]=labelencoder_y.fit_transform(df.iloc[:,1].values)


# In[65]:


sns.pairplot(df.iloc[:,1:6])


# In[66]:


sns.pairplot(df.iloc[:,1:6], hue = 'diagnosis')


# In[72]:


df.head()


# In[74]:


df.iloc[:,1:32].corr()


# In[75]:


sns.heatmap(df.iloc[:,1:31].corr())


# In[82]:


plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt='0.0%')


# In[ ]:




