#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


loans = pd.read_csv('loan_data.csv')


# In[3]:


loans.info() #not.fully.paid is the outcome variable
#First thought, is what is spread of that variables and others in relation to it


# In[4]:


loans.head()


# In[5]:


loans.describe() #We're going to need to normalize the values, they are all over the place


# In[53]:


plt.figure(figsize = (10,6))
loans[loans['credit.policy'] == 1]['fico'].hist(alpha = 0.5, bins = 30, color = 'blue',
                                                label = "credit underwriting met")
loans[loans['credit.policy'] == 0]['fico'].hist(alpha = 0.5, bins = 30, color = 'red',
                                                label = "credit underwriting not met")
plt.legend(loc = 'best')
plt.show()


# In[54]:


plt.figure(figsize = (10,6))
loans[loans['credit.policy'] == 1]['not.fully.paid'].hist(alpha = 0.5, bins = 30, color = 'blue',
                                                label = "credit underwriting met")
loans[loans['credit.policy'] == 0]['not.fully.paid'].hist(alpha = 0.5, bins = 30, color = 'red',
                                                label = "credit underwriting not met")
plt.legend(loc = 'best')
plt.title("not.fully.paid comparison by outcome\n Not stacked, just overlapping")
plt.show()


# In[58]:


plt.figure(figsize = (12,8))
sns.countplot(x = 'purpose', data = loans, hue = 'not.fully.paid', palette = 'Set1')


# In[25]:


sns.jointplot(x = 'fico', y = 'int.rate', data = loans, color = 'purple', size = 10)


# In[55]:


sns.lmplot('fico', 'int.rate', data = loans, hue = 'credit.policy', col = 'not.fully.paid', palette = 'Set1')


# In[27]:


loans.info()


# In[28]:


cat_feats = ['purpose']


# In[30]:


final_data.head()


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X = final_data.drop('not.fully.paid', axis = 1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


dtree = DecisionTreeClassifier()


# In[37]:


dtree.fit(X_train, y_train)


# In[38]:


pred = dtree.predict(X_test)


# In[39]:


from sklearn.metrics import classification_report, confusion_matrix


# In[41]:


print(classification_report(y_test, pred))


# In[42]:


print(confusion_matrix(y_test, pred))


# In[43]:


from sklearn.ensemble import RandomForestClassifier


# In[44]:


rfor = RandomForestClassifier()


# In[45]:


rfor.fit(X_train, y_train)


# In[46]:


pred = rfor.predict(X_test)


# In[47]:


print(classification_report(y_test, pred))


# In[48]:


print(confusion_matrix(y_test, pred))

