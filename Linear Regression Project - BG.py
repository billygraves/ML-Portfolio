#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["patch.force_edgecolor"] = True
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv('Ecommerce Customers')


# In[3]:


customers.head()


# In[4]:


customers.describe()


# In[5]:


customers.info()


# In[11]:


sns.jointplot('Time on Website', 'Yearly Amount Spent', data = customers)


# In[281]:





# In[12]:


sns.jointplot('Time on App', 'Yearly Amount Spent', data = customers)


# In[14]:


sns.jointplot('Time on App', 'Yearly Amount Spent', data = customers, kind = 'hex')


# In[16]:


sns.pairplot(customers)


# In[17]:


sns.lmplot('Length of Membership', 'Yearly Amount Spent', data = customers)


# In[18]:


customers.columns


# In[33]:


X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


lm = LinearRegression()


# In[38]:


lm.fit(X_train, y_train)


# In[39]:


print(lm.coef_)


# In[41]:


predictions = lm.predict(X_test)


# In[42]:


plt.scatter(y_test, predictions) #seems to fit a very good relationship


# In[46]:


from sklearn import metrics
print("MAE: ", metrics.mean_absolute_error(y_test, predictions))
print("MSE: ", metrics.mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[50]:


sns.distplot((y_test - predictions))
plt.title("Distribution of Residuals")


# In[53]:


pd.DataFrame(lm.coef_.transpose(), X_train.columns,
            columns = ['Coeffecient'])

