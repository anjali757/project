#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt


# In[2]:


data=pd.read_excel(r"C:\Users\Anjali Rajora\Downloads\Default.xlsx")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.describe(include="all")


# In[9]:


(data.balance==0).sum(axis=0)


# In[10]:


data.student.value_counts()


# In[11]:


data.default.value_counts()


# In[12]:


data['default2']=data.default.factorize()[0]


# In[13]:


data['student2']=data.student.factorize()[0]


# In[14]:


data.head()


# In[15]:


data_dfno = data[data.default2 == 0].sample(frac=0.15)
data_dfno


# In[16]:


data_dfyes = data[data.default2 == 1] 
data_dfyes


# In[17]:


data_df = data_dfno.append(data_dfyes)
data_df


# In[18]:


fig = plt.figure(figsize=(12,5)) 
gs = mpl.gridspec.GridSpec(1, 4) 
ax1 = plt.subplot(gs[0,:2]) 
ax2 = plt.subplot(gs[0,2:3]) 
ax3 = plt.subplot(gs[0,3:4]) 
ax1.scatter(data_df[data_df.default == 'Yes'].balance, data_df[data_df.default == 'Yes'].income, s=40, c='orange', marker='+', linewidths=1) 
ax1.scatter(data_df[data_df.default == 'No'].balance, data_df[data_df.default == 'No'].income, s=40, marker='o', linewidths=1, edgecolors='lightblue', facecolors='white', alpha=.6) 
ax1.set_ylim(ymin=0) 
ax1.set_ylabel('Income') 
ax1.set_xlim(xmin=-100) 
ax1.set_xlabel('Balance') 
c_palette = {'No':'lightblue', 'Yes':'orange'} 
sns.boxplot('default', 'balance', data=data, orient='v', ax=ax2, palette=c_palette) 
sns.boxplot('default', 'income', data=data, orient='v', ax=ax3, palette=c_palette) 
gs.tight_layout(plt.gcf())   


# In[19]:


get_ipython().system('python --version')


# In[20]:


import sklearn.linear_model as skl_lm


# In[21]:


clf = skl_lm.LogisticRegression(solver='newton-cg')


# In[22]:


X_train = data.balance.values.reshape(-1,1)        
y = data.default2
X_test = np.arange(data.balance.min(), data.balance.max()).reshape(-1 ,1)


# In[23]:


clf.fit(X_train,y)                      


# In[24]:


clf.fit(X_train,y)                      


# In[25]:


prob = clf.predict_proba(X_test)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5)) 
sns.regplot(data.balance, data.default2, order=1, ci=None,scatter_kws={'color':'orange'},line_kws={'color':'lightblue', 'lw':2}, ax=ax1) 
ax2.scatter(X_train, y, color='orange') 
ax2.plot(X_test, prob[:,1], color='lightblue') 
for ax in fig.axes: ax.hlines(1, xmin=ax.xaxis.get_data_interval()[0],xmax=ax.xaxis.get_data_interval()[1], linestyles='dashed', lw=1) 
ax.hlines(0, xmin=ax.xaxis.get_data_interval()[0],xmax=ax.xaxis.get_data_interval()[1], linestyles='dashed', lw=1) 
ax.set_ylabel('Probability of default') 
ax.set_xlabel('Balance') 
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.]) 
ax.set_xlim(xmin=-100) 


# In[26]:


print(clf)


# In[27]:


print('classes: ',clf.classes_)


# In[28]:


print('coefficients: ',clf.coef_)


# In[29]:


print('intercept :', clf.intercept_)


# In[30]:


import statsmodels.api as sm


# In[31]:


import statsmodels.discrete.discrete_model as sms


# In[34]:


X_train = sm.add_constant(data.balance)


# In[35]:


est = sm.Logit(y.ravel(), X_train).fit()


# In[36]:


est.summary2().tables[1]


# In[38]:


X_train = sm.add_constant(data.student2)


# In[39]:


y = data.default2


# In[40]:


est = sms.Logit(y, X_train).fit()


# In[41]:


print(est.summary().tables[1].as_text())


# In[42]:


X_train = sm.add_constant(data[['balance', 'income', 'student2']])


# In[43]:


est = sms.Logit(y, X_train).fit() 


# In[44]:


print(est.summary().tables[1])


# In[45]:


X_train = data[data.student == 'Yes'].balance.values.reshape(-1,1)


# In[46]:


y = data[data.student == 'Yes'].default2


# In[47]:


X_train2 = data[data.student == 'No'].balance.values.reshape(-1,1)

y2 = data[data.student == 'No'].default2


# In[48]:


X_test = np.arange(data.balance.min(), data.balance.max()).reshape(-1,1)


# In[52]:


clf = skl_lm.LogisticRegression(solver='newton-cg')



# In[51]:


clf2 = skl_lm.LogisticRegression(solver='newton-cg')


# In[53]:


clf.fit(X_train,y)


# In[54]:


clf2.fit(X_train2,y2) 


# In[55]:


prob = clf.predict_proba(X_test)    


# In[56]:


prob2 = clf2.predict_proba(X_test)


# In[57]:


data.groupby(['student','default']).size().unstack('default')


# In[59]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5)) 
# Left plot 
ax1.plot(X_test, prob[:,1], color='orange', label='Student') 
ax1.plot(X_test, prob2[:,1], color='lightblue', label='Non-student') 
ax1.hlines(127/2817, colors='orange', label='Overall Student',xmin=ax1.xaxis.get_data_interval()[0],xmax=ax1.xaxis.get_data_interval()[1], linestyles='dashed') 
ax1.hlines(206/6850, colors='lightblue', label='Overall Non-Student',xmin=ax1.xaxis.get_data_interval()[0],xmax=ax1.xaxis.get_data_interval()[1], linestyles='dashed') 
ax1.set_ylabel('Default Rate') 
ax1.set_xlabel('Credit Card Balance') 
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.]) 
ax1.set_xlim(450,2500) 
ax1.legend(loc=2) 
# Right plot 
sns.boxplot('student', 'balance', data=data, orient='v', ax=ax2,  palette=c_palette);


# In[ ]:




