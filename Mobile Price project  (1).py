#!/usr/bin/env python
# coding: utf-8

# # Mobile Price
# why the mobile prices is so expensive?

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('train.csv')
df.head()


# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df=pd.read_csv('train.csv')
df.head(20)


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df['ram'].isnull().sum()


# In[8]:


#we select numeric columns in the dataframe
numeric = df.select_dtypes(include=np.number)
numeric_columns = numeric.columns


# In[9]:


numeric


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


sns.factorplot('battery_power', data=df, kind='count')


# In[12]:


sns.factorplot('int_memory', data=df, hue= 'price_range', kind='count')


# In[13]:


df['talk_time'].hist(bins=70)


# In[14]:


df['int_memory'].hist(bins=8)


# In[16]:


x = weather_data["battery_power"]
y = weather_data["price_range"]

plt.scatter(x,y)
plt.xlabel("battery_power")
plt.ylabel("price_range")


# In[17]:


from sklearn.linear_model import LinearRegression

x = weather_data["battery_power"].values.reshape(-1,1)
y = weather_data["int_memory"]

lr_model = LinearRegression()
lr_model.fit(x, y)
y_pred = lr_model.predict(x)

plt.scatter(x,y)
plt.xlabel("battery_power")
plt.ylabel("int_memory")

plt.plot(x, y_pred)


# In[ ]:


theta_0 = lr_model.intercept_
theta_1 = lr_model.coef_
theta_0, theta_1


# In[18]:


y_pred = lr_model.predict(np.array([32]). reshape(1,1))
y_pred


# In[ ]:




