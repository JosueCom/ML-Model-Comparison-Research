#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries and data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Precision, Recall

import itertools

df = pd.read_csv("adjusted.csv")
X = np.array(df.drop(columns=[' Class']))
Y = np.array(df.iloc[:, -1])


# In[2]:


#give the various metrics based on actual and predicted data returned as a list
def metrics_(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    specificity = tn/(tn+fp)
    
    return [accuracy, sensitivity, specificity, precision]

def two_probability_predictions_as_binary(xTest, guesses):
    guesses_total = []
    for i in range(len(xTest)):
        benign = np.asscalar(guesses[i][0])
        malignant = np.asscalar(guesses[i][1])
        if benign > malignant:
            guesses_total.append(0)
        else:
            guesses_total.append(1)
    return guesses_total

def one_probability_predictions_as_binary(xTest, guesses):
    guesses_total = []
    for i in range(len(xTest)):
        probability = np.asscalar(guesses[i][0])
        if probability >= .5:
            guesses_total.append(1)
        else:
            guesses_total.append(0)
    return guesses_total

#tensorflow 4 layer model predictions
def deep_learning_4_predictions(xTrain, yTrain, second_layer_length, third_layer_length):
    model = Sequential([
    Flatten(input_shape=np.shape(xTrain[0])),
    Dense(second_layer_length, activation=tf.nn.relu),
    Dense(third_layer_length, activation=tf.nn.relu),
    Dense(len(np.unique(yTrain)), activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(xTrain, yTrain, epochs=100)
    
    
    guesses = model.predict(xTest)
    return two_probability_predictions_as_binary(xTest, guesses)

#tensorflow 3 layer model predictions
def deep_learning_3_predictions(xTrain, yTrain, second_layer_length):
    model = Sequential([
    Flatten(input_shape=np.shape(xTrain[0])),
    Dense(second_layer_length, activation=tf.nn.relu),
    #output is of shape 2
    Dense(len(np.unique(yTrain)), activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(xTrain, yTrain, epochs=100)

    #return the guesses as an array (tensorflow makes them as probabilities but we need only 1's and 0's)
    guesses = model.predict(xTest)
    return two_probability_predictions_as_binary(xTest, guesses)

def custom_model_predictions(xTrain, yTrain, second_layer_length, third_layer_length, dropout):
    #basically the shape of x
    input_size = 10
    data_input = Input(shape=(input_size,))
    hidden = Dense(second_layer_length, activation="softplus")(data_input)
    hidden = Dense(third_layer_length, activation="softmax")(hidden)
    hidden = Dropout(dropout)(hidden)
    #noutput is of shape 1
    data_output = Dense(1, activation="sigmoid")(hidden)
    model = Model(inputs=data_input, outputs=data_output)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
    model.fit(xTrain, yTrain, epochs=100)
    
    guesses = model.predict(xTest)
    return one_probability_predictions_as_binary(xTest, guesses)


# In[ ]:


#open "Rows: Acc Sens Spec Prec; Cols: Random_State Avg Standard_Dev.csv", & put into pandas df
df_metrics = pd.read_csv("Rows- Acc Sens Spec Prec; Cols- Random_State Avg Standard_Dev.csv")
#set collumns equal to zero before running, except for the 'row names' collumn.
for col in df_metrics.columns:
   if col != 'Name':
       df_metrics[col].values[:] = 0
#from random state 0 through 9 (inclusive), add to data frame we will use for the csv
for i in range(10):
   random_state = i
   xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.1, random_state=random_state)
   yTrain, yTest = yTrain[:,np.newaxis], yTest[:,np.newaxis]
   
   
   #various machine learning models
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
   
   #voting classifier based on machine learning models
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
   
   #various deep learning models
   #NN 3 layer 300
   guesses_3_300 = deep_learning_3_predictions(xTrain, yTrain, 300)
   #NN 4 layer 150 110
   guesses_4_150_110 = deep_learning_4_predictions(xTrain, yTrain, 150, 110)
   #NN 3 layer 200
   guesses_3_200 = deep_learning_3_predictions(xTrain, yTrain, 200)
   #NN 3 layer 180
   guesses_3_180 = deep_learning_3_predictions(xTrain, yTrain, 180)
   #NN 3 layer 160
   guesses_3_160 = deep_learning_3_predictions(xTrain, yTrain, 160)
   #NN 3 layer 120
   guesses_3_120 = deep_learning_3_predictions(xTrain, yTrain, 120)
   #NN 3 layer 80
   guesses_3_80 = deep_learning_3_predictions(xTrain, yTrain, 80)
   #custom model
   guesses_custom = custom_model_predictions(xTrain, yTrain, 20, 10, .2)
   #voting neural net
       #WILL DO LATER IF WE HAVE TIME
   
   #turn all the metrics into a single list
   rand_state_collumn = list(itertools.chain(metrics_(yTest, lr_guesses), 
                                             metrics_(yTest, svc_w_guesses),
                                             metrics_(yTest, svc_wo_guesses),
                                             metrics_(yTest, rf_guesses),
                                             metrics_(yTest, knn_one_guesses), 
                                             metrics_(yTest, naive_guesses),
                                             metrics_(yTest, decision_tree_guesses),
                                             metrics_(yTest, eclf_guesses),
                                             metrics_(yTest, guesses_3_300), 
                                             metrics_(yTest, guesses_4_150_110),
                                             metrics_(yTest, guesses_3_200), 
                                             metrics_(yTest, guesses_3_180),
                                             metrics_(yTest, guesses_3_160), 
                                             metrics_(yTest, guesses_3_120),
                                             metrics_(yTest, guesses_3_80), 
                                             metrics_(yTest, guesses_custom)))
                                               
   #add the various models as a collumn (based on random state) to the df
   df_metrics[str(i)] = rand_state_collumn
   
   #compute average of the various collumns and add as another collumn named avg
   df_metrics['Avg'] = df_metrics['Avg'] + df_metrics[str(i)]

   
   
df_metrics['Avg'] = df_metrics['Avg'] / 10    


#send df to csv
df_metrics.to_csv('Rows- Acc Sens Spec Prec; Cols- Random_State Avg Standard_Dev.csv')


# In[ ]:





# In[ ]:



