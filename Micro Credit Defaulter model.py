#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV








# In[2]:


df=pd.read_csv('micro credit loan.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df['msisdn'].describe()


# In[6]:


# we used the 'top' values with label showing 1
df[:][df['msisdn']=='04581I85330']


# # Data cleaning and EDA

# checking the missing value

# In[7]:


df.isnull().sum()


# In[8]:


df.shape


# comparison between the parameters amount spent, average main account balance, average payback

# In[9]:


df.corr()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (15,7))
sns.heatmap(df.corr(), annot=True, linewidths=0.5, linecolor='black', fmt='.2f')


# In[11]:


# Drop the index column
df.drop('Unnamed: 0',axis=1,inplace=True)


# In[12]:


# Drop "pcircle" column
df.drop("pcircle",axis=1,inplace=True)


# In[13]:


# Drop "pdate" column
df.drop("pdate",axis=1,inplace=True)


# In[14]:


# Drop "msisdn" column
df.drop("msisdn",axis=1,inplace=True)


# In[15]:


df.shape


# In[16]:


df.aon.plot(kind='density')


# In[17]:


# Target Variable (Label)
df_label = df.iloc[:,0]


# In[18]:


# Drop the target variable from dataset
df.drop("label",axis=1,inplace=True)


# In[19]:


df.shape


# In[20]:


df.head()


# # The selection using various classifiers and the visualization process

# In[21]:


#Concatenation process both NUMERIC and CATEGORICAL variables
df = pd.concat([df,df_label], axis=1)
df.head(5)


# In[22]:


X = df.drop(labels=['label'],axis=1)
Y = df.iloc[:,-1]


# In[23]:


X


# In[24]:


Y


# In[25]:


from sklearn.ensemble import RandomForestClassifier  as rf
model=rf()
fit = model.fit(X,Y)
score=model.feature_importances_


# In[26]:


from sklearn.feature_selection import RFE

rfe=RFE(model,25) #Selecting top 25 important features
fit=rfe.fit(X,Y)
results=fit.transform(X)
print(fit.n_features_)
print(fit.support_)
print(fit.ranking_)


# In[27]:


#Daily amount spent from main account, averaged over last 90 days (in Indonesian Rupiah)
plt.figure(figsize=(10,6))
ax = sns.boxplot(Y, X['daily_decr90'])
ax.set_title('Effect of Daily amount spent over last 90 days on Delinquency', fontsize=18)
ax.set_ylabel('Daily amount spent over last 90 days(in Indonesian Rupiah)', fontsize = 15)
ax.set_xlabel('Delinquency', fontsize = 15)


# In[28]:


#Daily amount spent from main account, averaged over last 30 days (in Indonesian Rupiah)
plt.figure(figsize=(10,6))
ax = sns.boxplot(Y, X['daily_decr30'])
ax.set_title('Effect of Daily amount spent over last 30 days on Delinquency', fontsize=18)
ax.set_ylabel('Daily amount spent over last 30 days(in Indonesian Rupiah)', fontsize = 15)
ax.set_xlabel('Delinquency', fontsize = 15)


# In[29]:


#Average main account balance over last 90 days
plt.figure(figsize=(10,6))
ax = sns.boxplot(Y, X['rental90'])
ax.set_title('Effect of Average balance over last 90 days on Delinquency', fontsize=18)
ax.set_ylabel('Avg balance over last 90 days', fontsize = 15)
ax.set_xlabel('Delinquency', fontsize = 15)


# In[30]:


#Average main account balance over last 30 days
plt.figure(figsize=(10,6))
ax = sns.boxplot(Y, X['rental30'])
ax.set_title('Effect of Average balance over last 30 days on Delinquency', fontsize=18)
ax.set_ylabel('Avg balance over last 30 days', fontsize = 15)
ax.set_xlabel('Delinquency', fontsize = 15)


# In[31]:


#create seperate train and test splits for validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[32]:


from sklearn import metrics
#create function for validation and return accuracy and roc-auc score
def evaluate_model(model):
    model.fit(X_train,y_train)
    prediction_test = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, prediction_test)
    rocauc = metrics.roc_auc_score(y_test, prediction_test)
    return accuracy,rocauc,prediction_test


# In[33]:


from sklearn.linear_model import LogisticRegression

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
acc,rocauc,testpred_lr  = evaluate_model(lr)
print('Logistic Regression...')
Y_LRpred=lr.predict(X_test)
print(classification_report(Y_LRpred,y_test))


# In[34]:


cm_LR = confusion_matrix(y_test, Y_LRpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_LR,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[35]:


rf =RandomForestClassifier()
rf.fit(X_train,y_train)
acc,rocauc,testpred_rf  = evaluate_model(rf)
print('Random Forest...')
Y_RFpred=rf.predict(X_test)
print(classification_report(Y_RFpred,y_test))


# In[36]:


cm_RF = confusion_matrix(y_test, Y_RFpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_RF,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[37]:


from sklearn.tree import DecisionTreeClassifier
dt =DecisionTreeClassifier()
dt.fit(X_train,y_train)
acc,rocauc,testpred_dt = evaluate_model(dt)
print('Decision Tree Classifier...')
Y_DTpred=dt.predict(X_test)
print(classification_report(Y_DTpred,y_test))


# In[38]:


cm_DT = confusion_matrix(y_test, Y_DTpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_DT,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[39]:


#SUPPORT VECTOR MACHINE
from sklearn.svm import SVC
svc =DecisionTreeClassifier()
svc.fit(X_train,y_train)
acc,rocauc,testpred_svc = evaluate_model(svc)
print('Support vector classifier...')
Y_SVCpred=svc.predict(X_test)
print(classification_report(Y_SVCpred,y_test))


# In[40]:


cm_SVC = confusion_matrix(y_test, Y_SVCpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_SVC,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[41]:


from sklearn.naive_bayes import GaussianNB
gnb =DecisionTreeClassifier()
gnb.fit(X_train,y_train)
acc,rocauc,testpred_gnb = evaluate_model(gnb)
print('Gaussian Navie Bayes...')
Y_GNBpred=gnb.predict(X_test)
print(classification_report(Y_GNBpred,y_test))


# In[42]:


cm_GNB = confusion_matrix(y_test, Y_GNBpred)
_,ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm_GNB,annot=True,fmt="d")
ax.set_ylim(2,-0.1)


# In[43]:


import itertools
from sklearn.metrics import roc_auc_score


lst    = [lr,rf,dt,gnb,svc]

length = len(lst)
mods   = ['Logistic Regression','Random Forest Classifier','Decision Tree',"Naive Bayes",'SVM Classifier']

plt.style.use("dark_background")
fig = plt.figure(figsize=(12,16))
fig.set_facecolor("#F3F3F3")
for i,j,k in itertools.zip_longest(lst,range(length),mods) :
    qx = plt.subplot(4,3,j+1)
    probabilities = i.predict_proba(X_test)
    predictions   = i.predict(X_test)
    fpr,tpr,thresholds = roc_curve(y_test,probabilities[:,1])
    plt.plot(fpr,tpr,linestyle = "dotted",
             color = "royalblue",linewidth = 2,
             label = "AUC = " + str(np.around(roc_auc_score(y_test,predictions),3)))
    plt.plot([0,1],[0,1],linestyle = "dashed",
             color = "orangered",linewidth = 1.5)
    plt.fill_between(fpr,tpr,alpha = .4)
    plt.fill_between([0,1],[0,1],color = "k")
    plt.legend(loc = "lower right",
               prop = {"size" : 12})
    qx.set_facecolor("k")
    plt.grid(True,alpha = .15)
    plt.title(k,color = "b")
    plt.xticks(np.arange(0,1,.3))
    plt.yticks(np.arange(0,1,.3))


# # Conclusion

# The  model has an Accuracy of 0.91 in Randomforestclassifier and AUC_ROC score is 0.71

# In[ ]:




