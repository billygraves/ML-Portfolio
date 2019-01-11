#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('bank_note_data.csv')


# In[3]:


df.head()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


sns.countplot(x = 'Class', data = df) #remember countplots you dumb
# bitch, it isn't that hard


# In[16]:


sns.pairplot(df, hue = 'Class') #data is seperated so it should go well


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


scaler = StandardScaler()


# In[20]:


scaler.fit(df.drop('Class', axis = 1))


# In[22]:


df = pd.concat([pd.DataFrame(scaler.transform(df.drop('Class', axis = 1)), columns = df.columns[:-1]), df['Class']], axis = 1)


# In[25]:


df.head()


# In[26]:


y = df['Class']


# In[27]:


X = df.drop('Class', axis = 1)


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 101)


# In[31]:


import tensorflow as tf


# In[35]:


feat_col = [tf.feature_column.numeric_column(x) for x in X_train.columns]


# In[37]:


classifier = tf.estimator.DNNClassifier(
    hidden_units = [10,20,10], feature_columns = feat_col,
    n_classes = 2)


# In[40]:


in_dat = tf.estimator.inputs.pandas_input_fn(
    x = X_train, y = y_train,
    batch_size = 20, shuffle = True)


# In[41]:


classifier.train(input_fn = in_dat, steps = 500)


# In[42]:


pred_fn = tf.estimator.inputs.pandas_input_fn(
    x = X_test, batch_size = len(X_test), shuffle = False)


# In[43]:


predictions = list(classifier.predict(input_fn = pred_fn))


# In[44]:


final_pred = [x['class_ids'][0] for x in predictions]


# In[26]:





# In[45]:


from sklearn.metrics import confusion_matrix, classification_report


# In[46]:


print(confusion_matrix(y_test, final_pred))


# In[47]:


print(classification_report(y_test, final_pred))


# In[48]:


#comparing the model
from sklearn.ensemble import RandomForestClassifier


# In[49]:


rf = RandomForestClassifier()


# In[50]:


rf.fit(X_train, y_train)


# In[51]:


pred = rf.predict(X_test)


# In[52]:


print(classification_report(y_test, pred))


# In[53]:


print(confusion_matrix(y_test, pred))

