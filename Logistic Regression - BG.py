#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["patch.force_edgecolor"] = True
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ad_data = pd.read_csv('advertising.csv')


# In[3]:


ad_data.head()


# In[4]:


ad_data.info()


# In[5]:


ad_data.describe()


# In[11]:


sns.set_style('whitegrid')
sns.distplot(ad_data['Age'], kde = False, bins = 30)


# In[12]:


sns.jointplot(x = 'Age', y = 'Area Income', data = ad_data)


# In[15]:


sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, color = 'red', kind = 'kde')


# In[17]:


sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data, color = 'green')


# In[19]:


sns.pairplot(ad_data, hue = 'Clicked on Ad')


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# In[25]:


from sklearn.model_selection import train_test_split
ad_data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis = 1, inplace = True)


# In[26]:


y = ad_data['Clicked on Ad']
X = ad_data.drop('Clicked on Ad', axis = 1)


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[30]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()


# In[31]:


log_model.fit(X_train, y_train)


# In[32]:


predictions = log_model.predict(X_test)


# In[33]:


from sklearn.metrics import classification_report


# In[35]:


print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))

