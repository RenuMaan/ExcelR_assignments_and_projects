#!/usr/bin/env python
# coding: utf-8

# ## Assignment 4 : Linear Regression

# In[218]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[219]:


df1 = pd.read_csv('delivery_time.csv')
df1.head()


# In[220]:


df1.info


# In[221]:


df1.isnull().sum()


# In[222]:


df1.describe()


# In[223]:


plt.subplot(1,2,1)
plt.boxplot(x= df1['Delivery Time'], data = df1)
plt.xlabel('Delivery Time')
plt.ylabel('frequency')
plt.subplot(1,2,2)
plt.boxplot(x= df1['Sorting Time'], data = df1)
plt.xlabel('Sorting Time')
plt.show()


# In[224]:


plt.figure(figsize = (3,3))
sns.regplot(x=df1['Delivery Time'], y = df1['Sorting Time'])


# In[225]:


x =np.array((df1['Delivery Time'])).reshape((-1, 1))
y =np.array(df1['Sorting Time'])


# In[226]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)


# In[227]:


r_sq = lr.score(x, y)
print("R-square value =", r_sq)
print(f"intercept: {lr.intercept_}")
print(f"slope: {lr.coef_}")


# In[228]:


y_hat = lr.predict(x)
y_hat


# In[229]:


plt.figure(figsize = (4,4))
plt.plot(x,y_hat, color = 'red')
plt.scatter(x, y)


# In[230]:


plt.figure(figsize = (3,3))
sns.residplot(x=y, y=y_hat)
plt.ylabel("Residues")


# In[231]:


resid = y-y_hat
plt.figure(figsize = (2,2))
plt.hist(resid, bins = 7, edgecolor = 'black')


# **Problem 2: Year of experience and Salary**

# In[232]:


df2 = pd.read_csv("Salary_Data.csv")
df2.head()


# In[233]:


df2.info()


# In[234]:


df2.describe()


# In[235]:


plt.subplots_adjust(wspace=0.4)
plt.subplot(1,2,1)
plt.boxplot(x= df2['YearsExperience'], data = df2)
plt.xlabel('YearsExperience')
plt.ylabel('frequency')
plt.subplot(1,2,2)
plt.boxplot(x= df2['Salary'], data = df2)
plt.xlabel('Salary')
plt.show()


# In[236]:


plt.figure(figsize = (3,3))
sns.regplot(x=df2['YearsExperience'], y = df2['Salary'])


# In[241]:


x =np.array((df2['YearsExperience'])).reshape((-1, 1))
y =np.array(df2['Salary'])


# In[242]:


lr1 = LinearRegression()
lr1.fit(x,y)
r_sq1 = lr1.score(x, y)
print("R-square value =", r_sq1)
print(f"intercept: {lr1.intercept_}")
print(f"slope: {lr1.coef_}")


# In[245]:


Y_hat1 = lr1.predict(x)
plt.figure(figsize = (4,4))
plt.plot(x,Y_hat1, color = 'red')
plt.scatter(x, y)


# In[247]:


plt.figure(figsize = (3,3))
sns.residplot(x=y, y=Y_hat1)
plt.ylabel("Residues")


# In[253]:


resid = y-Y_hat1
plt.figure(figsize = (2,2))
plt.hist(resid, bins = 8, edgecolor = 'black')


# In[ ]:




