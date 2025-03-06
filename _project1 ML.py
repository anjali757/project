#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns 
from matplotlib import pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
data=pd.read_csv(r"C:\Users\Anjali Rajora\Downloads\creditcard.csv")
data.head()


# In[2]:


data.describe()


# In[3]:


set_option('display.width',100)
data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# hence there is no null values in the data.

# In[7]:


class_names = {0:'Not Fraud', 1:'Fraud'}
print(data.Class.value_counts().rename(index = class_names))


# data cleansing is not required as the data is already cleaned (no null value is present in it for cleansing)

# In[8]:


from sklearn.model_selection import train_test_split
y=data["Class"]
x=data.loc[:,data.columns!='Class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=48,stratify=y)


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score#import libraries


# In[10]:


model = RandomForestClassifier() #initialise the classifier


# In[11]:


model.fit(x_train , y_train) #train model using dataset


# In[12]:


y_pred = model.predict(x_test) # prediction using test data


# In[13]:


acc=round(accuracy_score(y_test,y_pred)*100,2)
print("Accuracy of the model: ",acc) #accuracy by comparing y_test and y_pred


# random forest gives best accuracy. so now we can analyze confusion matrix for random forest

# In[14]:


from sklearn.metrics import confusion_matrix
con_matrix = confusion_matrix(y_test, y_pred)
con_matrix


# In[15]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, x_test, y_test)


# In[ ]:


'''In this case, overall accuracy is strong, but the confusion metrics tell a different story.
Despite the high accuracy level, 36 out of 164 instances of fraud are missed and incorrectly predicted as nonfraud.
The false-negative rate is substantial.
The intention of a fraud detection model is to minimize these false negatives.'''

