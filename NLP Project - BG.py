#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


yelp = pd.read_csv('yelp.csv')


# In[26]:


yelp.head()


# In[27]:


yelp.info()


# In[28]:


yelp.describe()


# In[29]:


yelp['text length'] = yelp['text'].apply(lambda x: len(x.split()))
yelp.head()


# In[30]:


#Already done!


# In[33]:


g = sns.FacetGrid(yelp, col = 'stars')
g.map(plt.hist, 'text length') #if it is established this way as a value it will work, have to establish for facet


# In[34]:


sns.boxplot(x = 'stars', y = 'text length', data = yelp)


# In[35]:


sns.countplot(yelp['stars'])


# In[36]:


yelp.groupby('stars')['cool', 'useful', 'funny', 'text length'].mean()


# In[39]:


yelp[['cool', 'useful', 'funny', 'text length']].corr()


# In[40]:


sns.heatmap(yelp[['cool', 'useful', 'funny', 'text length']].corr(), cmap = 'coolwarm')


# In[44]:


yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]


# In[45]:


X = yelp_class['text']
y = yelp_class['stars']


# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[47]:


X = cv.fit_transform(X)


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[51]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[52]:


nb.fit(X_train, y_train)


# In[53]:


pred = nb.predict(X_test)


# In[54]:


from sklearn.metrics import classification_report, confusion_matrix


# In[56]:


print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# In[57]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[58]:


from sklearn.pipeline import Pipeline


# In[59]:


pline = Pipeline([
    ('cv', CountVectorizer()),
    ('tfIdf', TfidfTransformer()),
    ('fit', MultinomialNB())
])


# In[65]:


X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[66]:


pline.fit(X_train, y_train)


# In[67]:


pred = pline.predict(X_test)


# In[68]:


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred)) #It's actually worse!

