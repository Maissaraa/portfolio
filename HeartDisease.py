#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os as os


# # Data Reading

# In[122]:


data=pd.read_csv('heart_2020_cleaned.csv')


# In[70]:


data.head(5)


# # Data Exploration And Visualization

# In[71]:


data.shape


# In[72]:


data.info()


# In[73]:


data.describe()


# In[74]:


data.dtypes


# In[75]:


data.isnull().sum()


# We can see that there are no missing values in the dataset. 

# In[76]:


#checking the frequency counts of variables.
for col in data.columns:
    
    print(f"""
###########################################################################################
The values of {col} is : 
{data[col].value_counts().sort_index()}
###########################################################################################""")


# ==> There are 17 variables in the dataset. All the variables are of categorical and numerical data type.
# 
# ==> Class "HeartDisease" is the target variable.
# 

# In[77]:


data.groupby("HeartDisease")['Sex'].value_counts(normalize=True)*100


# In[78]:


data.groupby("Smoking")['Sex'].value_counts(normalize=True)*100


# In[79]:


data.groupby("AlcoholDrinking")['Sex'].value_counts(normalize=True)*100


# In[80]:


encode_AgeCategory = {'55-59':57, '80 or older':80, '65-69':67,
                      '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                      '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                      '30-34':32,'25-29':27}
data['AgeCategory'] = data['AgeCategory'].apply(lambda x: encode_AgeCategory[x])
data['AgeCategory'] = data['AgeCategory'].astype('float')


# In[81]:


fig, ax = plt.subplots(figsize = (14,8))

sns.kdeplot(data[data["HeartDisease"]=='Yes']["AgeCategory"], alpha=1,shade = False, color="#ea4335", label="HeartDisease", ax = ax)
sns.kdeplot(data[data["KidneyDisease"]=='Yes']["AgeCategory"], alpha=1,shade = False, color="#4285f4", label="KidneyDisease", ax = ax)
sns.kdeplot(data[data["SkinCancer"]=='Yes']["AgeCategory"], alpha=1,shade = False, color="#fbbc05", label="SkinCancer", ax = ax)

ax.set_xlabel("AgeCategory")
ax.set_ylabel("Frequency")
ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.show()


# ==> People found to have heart disease, skin cancer & kidney disease are mostly old people

# In[82]:


fig, ax = plt.subplots(figsize = (14,8))

ax.hist(data[data["HeartDisease"]=='No']["Sex"], bins=3, alpha=0.8, color="#4285f4", label="No HeartDisease")
ax.hist(data[data["HeartDisease"]=='Yes']["Sex"], bins=3, alpha=1, color="#ea4335", label="HeartDisease")

ax.set_xlabel("Sex")
ax.set_ylabel("Frequency")

ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# In[83]:


female_with_heart_disease = len(data[(data['HeartDisease']=='Yes') & (data['Sex']=='Female')])
num_female = len(data[data['Sex']=='Female'])
male_with_heart_disease = len(data[(data['HeartDisease']=='Yes') & (data['Sex']=='Male')])
num_male = len(data[data['Sex']=='Male'])
print('Probability of Male to have Heart disease:', male_with_heart_disease/num_male)
print('Probability of Female to have Heart disease:', female_with_heart_disease/num_female)


# ==> Most heart disease patients are Male than Females
# 
# ==> More Females were tested than males
# 
# ==> Males are approximately 1.6 times more likely to have heart disease than females

# In[84]:


fig, ax = plt.subplots(figsize = (14,8))

ax.hist(data[data["HeartDisease"]=='No']["Smoking"], bins=3, alpha=0.8, color="#4285f4", label="No HeartDisease")
ax.hist(data[data["HeartDisease"]=='Yes']["Smoking"], bins=3, alpha=1, color="#ea4335", label="HeartDisease")

ax.set_xlabel("Smoking")
ax.set_ylabel("Frequency")

ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# In[85]:


smoke_and_heart_disease = len(data[(data['HeartDisease']=='Yes') & (data['Smoking']=='Yes')])
num_smoke = len(data[data['Smoking']=='Yes'])
no_smoke_and_heart_disease = len(data[(data['HeartDisease']=='Yes') & (data['Smoking']=='No')])
num_no_smoke = len(data[data['Smoking']=='No'])
print('Probability of Heart disease if you smoke:', smoke_and_heart_disease/num_smoke)
print("Probability of Heart disease if you don't smoke:", no_smoke_and_heart_disease/num_no_smoke)


# ==> Most heart disease patients smoke
# 
# ==> People who smoke are approximately twice as likely to have heart disease than people who don't smoke

# In[86]:


fig, ax = plt.subplots(figsize = (14,8))

ax.hist(data[data["HeartDisease"]=='No']["Race"], bins=15, alpha=0.8, color="#4285f4", label="No HeartDisease")
ax.hist(data[data["HeartDisease"]=='Yes']["Race"], bins=15, alpha=1, color="#ea4335", label="HeartDisease")

ax.set_xlabel("Race")
ax.set_ylabel("Frequency")

ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# ==> Most Heart Disease Patients are White people

# In[87]:


fig, ax = plt.subplots(figsize = (10,8))

ax.hist(data[data["HeartDisease"]=='No']["GenHealth"], bins=10, alpha=0.8, color="#4285f4", label="No HeartDisease")
ax.hist(data[data["HeartDisease"]=='Yes']["GenHealth"], bins=10, alpha=1, color="#ea4335", label="HeartDisease")

ax.set_xlabel("GenHealth")
ax.set_ylabel("Frequency")

ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# In[88]:


data["HeartDisease"].value_counts().plot.pie()


# In[89]:


# classification of BMI (Body Mass Index) by ranges :
BMI_UnderWeight = data['BMI'][(data['BMI']>=0) & (data['BMI'] <= 18.5)]
BMI_NormalRange = data['BMI'][(data['BMI']>18.5) & (data['BMI'] <= 25)]
BMI_OverrWeight = data['BMI'][(data['BMI']>25) & (data['BMI'] <= 30)]
BMI_Obese = data['BMI'][(data['BMI']>30)]


# In[90]:


BMI_X = ['UnderWeight','NormalRange','OverWeight','Obese']
BMI_Y = [len(BMI_UnderWeight.values),len(BMI_NormalRange.values),len(BMI_OverrWeight.values),len(BMI_Obese.values)]


# In[91]:


plt.figure(figsize = (14,6))
sns.barplot (x = BMI_X, y = BMI_Y)


# In[92]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="Smoking")


# In[93]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="AlcoholDrinking")


# In[94]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="Stroke")


# In[95]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="Sex")


# In[96]:


data['AgeCategory'].value_counts().sort_index()


# In[97]:


# classification of Age Category by ranges :
Young   = data['AgeCategory'][(data['AgeCategory'] <= 32.0)]
Mature  = data['AgeCategory'][(data['AgeCategory'] >= 37.0) & (data['AgeCategory'] <= 47.0)]
Senior  = data['AgeCategory'][(data['AgeCategory'] >= 52.0) & (data['AgeCategory'] <= 62.0)]
Old     = data['AgeCategory'][(data['AgeCategory'] >= 67.0) & (data['AgeCategory'] <= 77.0)]
Veryold = data['AgeCategory'][(data['AgeCategory'] == 80.0)]


# In[98]:


Age_X = ['Young','Mature','Senior','Old','Veryold']
Age_Y = [len(Young.values),len(Mature.values),len(Senior.values),len(Old.values),len(Veryold.values)]


# In[99]:


plt.figure(figsize = (14,8))
sns.barplot (x = Age_X, y = Age_Y)


# In[100]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="AgeCategory")
# 20          21064
# 27          16955
# 32          18753
# 37          20550
# 42          21006
# 47          21791
# 52          25382
# 57          29757
# 62          33686
# 67          34151
# 72          31065
# 77          21482
# 80          24153


# In[101]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="Race")
# 0 ==> American Indian/Alaskan Native      5202
# 1 ==> Asian                               8068
# 2 ==> Black                              22939
# 3 ==> Hispanic                           27446
# 4 ==> Other                              10928
# 5 ==> White                              245212


# In[102]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="Diabetic")
# 0 ==> No                               5202
# 1 ==> No, borderline diabetes          8068
# 2 ==> Yes                              22939
# 3 ==> Yes (during pregnancy)           27446


# In[103]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="PhysicalActivity")
# 0 ==> No      71838
# 1 ==> Yes    247957


# In[104]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="GenHealth")
#0 ==> Excellent     66842
#1 ==> Fair          34677
#2 ==> Good          93129
#3 ==> Poor          11289
#4 ==> Very good    113858


# In[105]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="SleepTime")


# In[106]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="Asthma")
#الربو


# In[107]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="KidneyDisease")


# In[108]:


plt.figure(figsize = (14,8))
sns.countplot(data=data,x="SkinCancer")


# In[109]:


plt.figure(1,figsize=(20,10))
sns.heatmap(data.corr(),annot=True)
plt.show()


# In[110]:


data.hist(bins=50, figsize=(20,15))
plt.show()


# # Model Selection

# In[111]:


#We are going to increase the minority class
#import imblearn
#from imblearn.over_sampling  import RandomOverSampler
#smote = RandomOverSampler(sampling_strategy=1)
#x_smote, y_smote = smote.fit_resample(x,y)


# In[118]:


#y_smote.value_counts().plot.pie()


# In[123]:


from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import preprocessing


#x = df[['BMI','PhysicalHealth','MentalHealth','SleepTime']]
X = data[['BMI','PhysicalHealth','MentalHealth','SleepTime']]
y = data["HeartDisease"]

# normalization of the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)
 
# Train-and-Test -Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[124]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[125]:


# Confusion matrix
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
 
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm,
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (8, 5))
sn.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens")
plt.show()
 
print('The details for confusion matrix is =')
print (classification_report(y_test, y_pred))


# In[126]:


CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)


# In[ ]:





# In[ ]:




