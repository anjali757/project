#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv(r"C:\Users\Anjali Rajora\Downloads\diabetes-data.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data['Outcome'].value_counts()


# In[6]:


data.groupby('Outcome').mean()


# In[7]:


data.describe()


# In[8]:


data.describe().T


# In[9]:


data_copy=data.copy(deep=True)
data_copy[['Glucose','BloodPressure','SkinThickness','BMI','Insulin']]=data_copy[['Glucose','BloodPressure','SkinThickness','BMI','Insulin']].replace(0,np.NAN)


# In[10]:


print(data_copy.isnull().sum())


# In[11]:


hplot=data.hist(figsize=(40,40))


# In[12]:


data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace=True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace=True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace=True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace=True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace=True)


# In[13]:


hplot=data_copy.hist(figsize=(40,40))


# In[14]:


print(data.Outcome.value_counts())


# In[15]:


p=data.Outcome.value_counts().plot(kind="bar")


# outcome variable is categorical and the non diabetes category(0) is approx twice the diabetes category(1)

# In[16]:


from pandas.plotting import scatter_matrix #for unclean data
p=scatter_matrix(data,figsize=(25, 25)) #scatterplot gives the relationship btw two variables


# In[17]:


sns.pairplot(data_copy,hue="Outcome") #of clean data


# In[18]:


plt.figure(figsize=(10,10))
p=sns.heatmap(data.corr(),annot=True,cmap='RdYlGn') #to visualise correlation matrix


# In[19]:


plt.figure(figsize=(10,10))
p=sns.heatmap(data_copy.corr(),annot=True,cmap='RdYlGn') #for clean data


# highest correlation btw glucose and outcome(0.49) and lowest correlation btw bloodpressure and outcome(0.17)

# In[20]:


from sklearn.preprocessing import StandardScaler
xscale = StandardScaler()
X = xscale.fit_transform(data_copy.drop(["Outcome"],axis = 1),)
X = pd.DataFrame(X,columns=['Pregnancies', 'Glucose', 
'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
'DiabetesPedigreeFunction', 'Age'])
X.head()


# In[21]:


Y = data_copy.Outcome
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2, stratify=Y)


# In[22]:


print(Y)


# In[23]:


scaler = StandardScaler()


# In[24]:


scaler.fit(X)


# In[25]:


standardized_data = scaler.transform(X)


# In[26]:


print(standardized_data)


# In[27]:


X= standardized_data
Y=data['Outcome']


# In[28]:


print(X)
print(Y)


# In[29]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, stratify= Y,random_state =2)


# In[30]:


print(X.shape,X_train.shape,X_test.shape)


# In[31]:


classifier = svm.SVC(kernel = 'linear')


# In[32]:


classifier.fit(X_train,Y_train)


# In[33]:


X_train_prediction= classifier.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction, Y_train)


# In[34]:


print("accuracy score of training data:",training_data_accuracy)


# In[35]:


X_test_prediction= classifier.predict(X_test)
testing_data_accuracy= accuracy_score(X_test_prediction, Y_test)


# In[36]:


print("accuracy score of testing data:",testing_data_accuracy)


# #make a predictive system
# 

# In[40]:


input_data=(4,110,92,0,37.6,0,191,30)

#convert input data to numpy array
numpy_arr=np.asarray(input_data)

#reshape array
input_data_reshape = numpy_arr.reshape(1,-1)

#standardize the input data
std_data=scaler.transform(input_data_reshape)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

if(prediction[0] == 0):
    print('person is not diabetic')
else:
    print('person is diabetic')


# In[ ]:




