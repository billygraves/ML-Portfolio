#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('KNN_Project_Data') #Not real data


# In[3]:


df.head()


# In[4]:


sns.pairplot(df, hue = 'TARGET CLASS')


# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


scale = StandardScaler()


# In[9]:


scale.fit(df.drop('TARGET CLASS', axis = 1))


# In[11]:


scaled_vars = scale.transform(df.drop('TARGET CLASS', axis = 1))


# In[15]:


scaled_df = pd.DataFrame(scaled_vars, columns = df.columns[:-1])
scaled_df.head()


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(scaled_df, df['TARGET CLASS'], test_size = 0.3, random_state = 42)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier


# In[19]:


knn = KNeighborsClassifier(n_neighbors= 1)


# In[20]:


knn.fit(X_train, y_train)


# In[21]:


pred = knn.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix


# In[23]:


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[18]:





# In[27]:


error_rate = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rate.append((pred != y_test).mean())


# In[42]:


plt.figure(figsize = (12,8))
plt.plot(error_rate, ls = '--', marker = 'o', color = 'blue',
         markerfacecolor = 'red', markeredgecolor = 'black', markersize = 10)
print(error_rate.index(min(error_rate)))


# In[41]:


knn = KNeighborsClassifier(n_neighbors = 20)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

