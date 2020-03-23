#!/usr/bin/env python
# coding: utf-8

# In[157]:


import pandas as pd  
import numpy as np


# In[158]:


titanic = pd.read_csv('/Users/Niklas/Desktop/Master ESCP/ML Python/titanic_data_ESCP.csv')


# In[159]:


titanic


# In[160]:


## 1. What is the total number of people on the titanic? How many of them survived and how many did not?
titanic.PassengerId.count()


# In[161]:


PeopleS = titanic[(titanic["Survived"] == 1)]
PeopleS.Survived.count()


# In[162]:


PeopleN = titanic[(titanic["Survived"] == 0)]
PeopleN.Survived.count()


# In[163]:


## 2. How many that survived were female and how many that died were female?


# In[164]:


PeopleSF = titanic[(titanic["Survived"] == 1) & (titanic["Sex"] == "female")]
PeopleSF.Survived.count()


# In[165]:


PeopleNF = titanic[(titanic["Survived"] == 0) & (titanic["Sex"] == "female")]
PeopleNF.Survived.count()


# In[166]:


## 3. How many children were on the titanic? NB: you are a child if age < 17


# In[167]:


PeopleC = titanic[(titanic["Age"] < 17)]
PeopleC.PassengerId.count()


# In[168]:


## 4. How many children died that were on the ship?


# In[169]:


PeopleCD = titanic[(titanic["Survived"] == 0) & (titanic["Age"] < 17)]
PeopleCD.PassengerId.count()


# In[170]:


## 5. How many people had families with them?


# In[171]:


PeopleF = titanic[(titanic["Siblings - Spouse"] >= 1) | (titanic["Parents - Children"] <= 1)]
PeopleF.PassengerId.count()


# In[172]:


## 6. What is the ratio of female to male?


# In[173]:


Female = titanic[(titanic["Sex"] == "female")]
Male = titanic[(titanic["Sex"] == "male")]
(Female.PassengerId.count())/(Male.PassengerId.count())


# In[174]:


## 7. What contributed to the survival of those who survived?


# In[175]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[176]:


## 7.1 Clean data, select relevant columns and prepare for model
titanic1 = titanic.drop(["PassengerId","Name","Ticket","Cabin"], axis = 1)
titanic1


# In[186]:


## One hot encode Sex and Port of embarkation column
titanic1 = pd.get_dummies(titanic1, columns=["Sex","Port of embarkation"])


# In[187]:


titanic1


# In[195]:


titanic1.isnull().sum()


# In[197]:


titanic1["Age"].fillna(titanic1["Age"].mean(), inplace = True)


# In[199]:


titanic1.isnull().sum()


# In[201]:


titanic1 = titanic1.dropna()


# In[202]:


titanic1.isnull().sum()


# In[208]:


## 7.2 Split dataset into train and test data
X = titanic1.drop("Survived", axis = 1)
Y = titanic1.Survived

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


# In[211]:


import statsmodels.api as sm


# In[212]:


logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())


# In[205]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[206]:


logreg = LogisticRegression(random_state=0)
logreg.fit(X_train, Y_train)


# In[207]:


Y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))


# In[214]:


from sklearn.metrics import confusion_matrix


# In[215]:


confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)


# In[216]:


from sklearn.metrics import classification_report


# In[217]:


print(classification_report(Y_test, Y_pred))


# In[218]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(Y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[219]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# In[220]:


rf = RandomForestClassifier(criterion='gini', n_estimators=700, min_samples_split=10, max_features='auto')
rf1 = RandomForestRegressor(n_estimators =1000, random_state=42)


# In[221]:


rand_for = rf.fit(X_train,Y_train)


# In[222]:


Importance = rand_for.feature_importances_


# In[223]:


std = np.std([tree.feature_importances_ for tree in rand_for.estimators_], axis=0)
indices = np.argsort(Importance)[::-1]


# In[224]:


print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], Importance[indices[f]]))


# In[226]:


# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), Importance[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[228]:


list(X_train.columns)
X_train.head(10)

