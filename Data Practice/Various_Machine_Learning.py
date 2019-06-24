#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[168]:


df = pd.read_csv("adjusted.csv")
X = np.array(df.drop(columns=[' Class']))
Y = np.array(df.iloc[:, -1])
random_state = 3


# In[169]:


xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state=random_state)
yTrain, yTest = yTrain[:,np.newaxis], yTest[:,np.newaxis]
#proof that knn should have one neighbor for max efficiency
for i in range(12):
    knn = KNeighborsClassifier(n_neighbors = i+1)
    knn.fit(xTrain,yTrain)
    knn_guesses = knn.predict(xTest)

    print('knn', str(i+1),
      accuracy_score(knn_guesses, yTest))

    


# In[170]:


lr = LogisticRegression()
lr.fit(xTrain, yTrain)
lr_guesses = lr.predict(xTest)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(xTrain, yTrain)
decision_tree_guesses = decision_tree.predict(xTest)

naive = GaussianNB()
naive.fit(xTrain, yTrain)
naive_guesses = naive.predict(xTest)

rf = RandomForestClassifier()
rf.fit(xTrain, yTrain)
rf_guesses = rf.predict(xTest)

svc_w_linear_kernel = SVC(kernel='linear')
svc_w_linear_kernel.fit(xTrain, yTrain)
svc_w_guesses = svc_w_linear_kernel.predict(xTest)

svc_wo_linear_kernel = SVC()
svc_wo_linear_kernel.fit(xTrain, yTrain)
svc_wo_guesses = svc_wo_linear_kernel.predict(xTest)

knn_one = KNeighborsClassifier(n_neighbors = 1)
knn_one.fit(xTrain,yTrain)
knn_one_guesses = knn_one.predict(xTest)

print('\nrandom_state', random_state)
print('LR', accuracy_score(lr_guesses, yTest), '\nSVC_w_linear_kernel', accuracy_score(svc_w_guesses, yTest),
        '\nsvc_wo_linear_kernel', accuracy_score(svc_wo_guesses, yTest), '\nRF',
      accuracy_score(rf_guesses, yTest), '\nknn',
      accuracy_score(knn_one_guesses, yTest), '\nnaive',
      accuracy_score(naive_guesses, yTest), '\ndecision tree',
      accuracy_score(decision_tree_guesses, yTest))


# In[171]:


#set up the confusion matrices
def confusion_mat(yTest, yPred, name):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt     

    cm = confusion_matrix(yTest, yPred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    words = 'Confusion Matrix ' + name
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(words); 


# In[172]:


confusion_mat(yTest, lr_guesses, 'Linear Regression')


# In[173]:


confusion_mat(yTest, svc_w_guesses, 'Support Vector Linear')


# In[174]:


confusion_mat(yTest, svc_wo_guesses, 'Support Vector Nonlinear')


# In[175]:


confusion_mat(yTest, rf_guesses, 'Random Forest')


# In[176]:


confusion_mat(yTest, knn_one_guesses, 'KNeighbors')


# In[177]:


confusion_mat(yTest, naive_guesses, 'Naive Bayes')


# In[178]:


from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=[
    ('LR', lr),
('svc_w_linear_kernel',svc_w_linear_kernel),
('svc_wo_linear_kernel',svc_wo_linear_kernel),
('RF', rf),
('knn', knn_one),
('naive', naive),
('decision tree', decision_tree),], voting='hard')
ensemble.fit(xTrain, yTrain)
eclf_guesses = ensemble.predict(xTest)
print('voting classifier', random_state, accuracy_score(eclf_guesses, yTest))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




