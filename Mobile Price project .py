#!/usr/bin/env python
# coding: utf-8

# # Mobile Price
# why the mobile prices is so expensive?

# In[1]:


import numpy as np
import pandas as pd


# In[7]:


df=pd.read_csv('train.csv')
df.head()


# In[3]:


import numpy as np
import pandas as pd


# In[8]:


df=pd.read_csv('train.csv')
df.head(20)


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[19]:


df['ram'].isnull().sum()


# In[ ]:




