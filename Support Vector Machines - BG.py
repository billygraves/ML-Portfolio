#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[2]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[3]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
iris = sns.load_dataset('iris')


# In[7]:


#Already done but I will take a look at the head of the data here
iris.head()


# In[8]:


sns.pairplot(iris, hue = 'species') #easiest to seperate is setosa


# In[21]:


sns.kdeplot(iris[iris['species'] == 'setosa'].drop(['petal_length', 'petal_width', 'species'], axis  =1),
           shade = True, cmap = 'plasma')
plt.xlabel(iris.columns[0])
plt.ylabel(iris.columns[1])


# In[23]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(iris.drop('species', axis = 1), iris['species'],
                                                    test_size = 0.3, random_state = 42)


# In[26]:


from sklearn.svm import SVC


# In[27]:


model = SVC()


# In[28]:


model.fit(X_train, y_train)


# In[29]:


pred = model.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report, confusion_matrix


# In[31]:


print(confusion_matrix(y_test, pred))


# In[32]:


print(classification_report(y_test, pred)) #The model is literally perfect


# In[33]:


from sklearn.grid_search import GridSearchCV


# In[35]:


param_grid = {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, .1, .01, .001, .0001]}


# In[37]:


search = GridSearchCV(SVC(), param_grid, verbose = 1)
search.fit(X_train, y_train)


# In[39]:


pred = search.predict(X_test)


# In[40]:


print(confusion_matrix(y_test, pred))


# In[41]:


print(classification_report(y_test, pred))

