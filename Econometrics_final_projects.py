#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np 
import pandas as pd 
from pandas import set_option
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import statsmodels.api as sm


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import performance metrics/measures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[2]:


SHOW_FIGURE = False 


# In[3]:


df = pd.read_csv('/Users/benson/Desktop/大四下/計量經濟/default of credit card clients.csv') 


# # Data cleaning

# In[4]:


# Drop "ID" column.
df = df.drop(['ID'], axis=1)
df = df.drop_duplicates()
df.info()


# In[5]:


df['MARRIAGE'].value_counts()


# In[6]:


df['EDUCATION'].value_counts()


# In[7]:


# category '0' undocumented is deleted
df = df.drop(df[df['MARRIAGE']==0].index)
# we could also group the 0 category with 3:others
# data['MARRIAGE']=np.where(data['MARRIAGE'] == 0, 3, data['MARRIAGE'])

# categories 0, 5 and 6 are unknown and are deleted
df = df.drop(df[df['EDUCATION']==0].index)
df = df.drop(df[df['EDUCATION']==5].index)
df = df.drop(df[df['EDUCATION']==6].index)


# In[8]:


font = FontProperties()
font.set_family(['Times New Roman', 'serif'])
font.set_size(14)
# 1=graduate school, 2=university, 3=high school 4=others
df['EDUCATION'].value_counts().plot(kind='bar', figsize=(10,6))
# plt.title("Number of cars by make")
plt.xticks([0,1,2,3],['University','Graduate\nSchool', 'High\nSchool', 'Others'],fontproperties=font,rotation=0)
# plt.xlabel('Education level', fontproperties=font)
plt.ylabel('# of clients', fontproperties=font)


# In[9]:


# 1=married, 2=single, 3=others
df['MARRIAGE'].value_counts().plot(kind='bar', figsize=(10,6))
# plt.title("Number of cars by make")
plt.xticks([0,1,2],['Single','Married', 'Others'],fontproperties=font,rotation=0)
# plt.xlabel('Marital Status', fontproperties=font)
plt.ylabel('# of clients', fontproperties=font)
plt.show()


# # Check collinearity

# In[10]:


plt.figure(figsize=(20,20))

sns.heatmap(df.corr(), annot=True, cmap='rainbow',linewidth=0.5, fmt='.2f')

# we see BILL_AMT has strong collinearity => remain BILL_AMT1 only!
# we also see that marraige2, sex2 has collinearity 


# In[11]:


# dropping PAY_2~6, marraige2, sex2
df.drop(['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], axis = 1, inplace = True)


# # Set 'category' type to categorical attributes, using one-hot encoding

# In[12]:


for att in ['SEX', 'EDUCATION', 'MARRIAGE']:
  df[att] = df[att].astype('category')

# one-hot encoding
df = pd.concat([pd.get_dummies(df['SEX'], prefix='SEX'),
                pd.get_dummies(df['EDUCATION'], prefix='EDUCATION'), 
                pd.get_dummies(df['MARRIAGE'], prefix='MARRIAGE'),
                df],axis=1)
# drop original columns
df.drop(['EDUCATION'],axis=1, inplace=True)
df.drop(['SEX'],axis=1, inplace=True)
df.drop(['MARRIAGE'],axis=1, inplace=True)

# dropping marraige2, sex2
df.drop(['SEX_2', 'MARRIAGE_2'], axis = 1, inplace = True)


# In[13]:


plt.figure(figsize=(17,15))

sns.heatmap(df.corr(), annot=True, cmap='rainbow',linewidth=0.5, fmt='.2f')


# In[14]:


df.drop(['EDUCATION_2'], axis = 1, inplace = True)


# # Check for null value

# In[15]:


df.isna().any().sum()


# In[16]:


df.info()


# # Divide data

# In[17]:


X = df[df.columns[:-1]]
y = df['dpnm']


# In[18]:


X.shape


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)


# In[20]:


# Add a constant column to the feature matrix
X_train = sm.add_constant(X_train)

# Create and fit the logistic regression model
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()

# Print the summary to view coefficient statistics, including p-values
print(result.summary())


# ## Select significant variables (p-value <0.05)

# In[21]:


new_X_train = X_train[['SEX_1', 'EDUCATION_1', 'EDUCATION_4', 'MARRIAGE_1', 'LIMIT_BAL',
                      'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'BILL_AMT1', 'PAY_AMT1', 'PAY_AMT2']]

new_X_test = X_test[['SEX_1', 'EDUCATION_1', 'EDUCATION_4', 'MARRIAGE_1', 'LIMIT_BAL',
                      'AGE', 'PAY_1', 'PAY_2', 'PAY_3', 'BILL_AMT1', 'PAY_AMT1', 'PAY_AMT2']]


# In[22]:


# Add a constant column to the feature matrix
new_X_train = sm.add_constant(new_X_train)

# Create and fit the logistic regression model
logit_model = sm.Logit(y_train, new_X_train)
new_result = logit_model.fit()

# Print the summary to view coefficient statistics, including p-values
print(new_result.summary())


# # Predict testing data

# In[23]:


# Add a constant column to the feature matrix of the test data
X_test = sm.add_constant(X_test)

# Make predictions on the test data
predictions = result.predict(X_test)

# Round the probabilities to get binary predictions (0 or 1)
binary_predictions = (predictions >= 0.5).astype(int)


# In[24]:


# Precision
print('Precision: %.3f' % precision_score(y_test, binary_predictions))

# Recall
print('Recall: %.3f' % recall_score(y_test, binary_predictions))


# f1 score: 
print('F1 score: %.3f' % f1_score(y_test, binary_predictions))


# In[25]:


# Plot confusion matrix for Logistic Regression.
logreg_matrix = confusion_matrix(y_test,binary_predictions)
sns.set(font_scale=1.3)
plt.subplots(figsize=(5, 5))
sns.heatmap(logreg_matrix, annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for Logistic Regression');


# ## Predict selected testing data

# In[26]:


# # Add a constant column to the feature matrix of the test data
new_X_test = sm.add_constant(new_X_test)

# Make predictions on the test data
predictions = new_result.predict(new_X_test)

# Round the probabilities to get binary predictions (0 or 1)
new_binary_predictions = (predictions >= 0.5).astype(int)

# Precision
print('Precision: %.3f' % precision_score(y_test, new_binary_predictions))

# Recall
print('Recall: %.3f' % recall_score(y_test, new_binary_predictions))


# f1 score: 
print('F1 score: %.3f' % f1_score(y_test, new_binary_predictions))


# In[27]:


# Plot confusion matrix for Logistic Regression.
logreg_matrix = confusion_matrix(y_test, new_binary_predictions)
sns.set(font_scale=1.3)
plt.subplots(figsize=(5, 5))
sns.heatmap(logreg_matrix, annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for selected Logistic Regression');

