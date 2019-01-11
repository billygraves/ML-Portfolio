#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df = pd.read_csv('College_Data', index_col = 0)


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


df.describe()


# In[20]:


sns.lmplot(x = 'Grad.Rate', y = 'Room.Board', data = df, hue = 'Private',
           fit_reg = False, aspect = 1, palette = 'coolwarm')


# In[21]:


sns.lmplot(x = 'F.Undergrad', y = 'Outstate', data = df, hue = 'Private',
          fit_reg = False, aspect = 1, palette = 'coolwarm')


# In[24]:


plt.figure(figsize = (12,8))
df[df['Private'] == 'Yes']['Outstate'].plot(kind = 'hist', color = 'blue', alpha = 0.6)
df[df['Private'] == 'No']['Outstate'].plot(kind = 'hist', color = 'red', alpha = 0.6)
plt.legend()


# In[25]:


plt.figure(figsize = (12,8))
df[df['Private'] == 'Yes']['Grad.Rate'].plot(kind = 'hist', color = 'blue', alpha = 0.6)
df[df['Private'] == 'No']['Grad.Rate'].plot(kind = 'hist', color = 'red', alpha = 0.6)
plt.legend()


# In[26]:


df[df['Grad.Rate'] > 100] #Cazenovia


# In[33]:


df.loc['Cazenovia College', 'Grad.Rate'] = 100


# In[34]:


print(df.loc['Cazenovia College', 'Grad.Rate'])


# In[35]:


plt.figure(figsize = (12,8))
df[df['Private'] == 'Yes']['Grad.Rate'].plot(kind = 'hist', color = 'blue', alpha = 0.6)
df[df['Private'] == 'No']['Grad.Rate'].plot(kind = 'hist', color = 'red', alpha = 0.6)
plt.legend()


# In[36]:


from sklearn.cluster import KMeans


# In[37]:


kmeans = KMeans(n_clusters = 2)


# In[38]:


kmeans.fit(df.drop('Private', axis = 1))


# In[39]:


kmeans.cluster_centers_


# In[43]:


def change(obj):
    if obj == 'Yes':
        return 1
    else:
        return 0


# In[44]:


df['Cluster'] = df['Private'].apply(change)


# In[45]:


df.head()


# In[48]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))

