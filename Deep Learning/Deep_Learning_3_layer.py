#!/usr/bin/env python
# coding: utf-8

# In[25]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("adjusted.csv")


# In[26]:


X = np.array(df.drop(columns=[' Class']))
Y = np.array(df.iloc[:, -1])
print(X.shape)
print(Y.shape)


# In[57]:


xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state = 3)
yTrain, yTest = yTrain[:,np.newaxis], yTest[:,np.newaxis] #add an extra input layer for the classification
print(xTrain.shape)
print(yTrain.shape)
print(len(xTrain[0]))


# In[69]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=np.shape(xTrain[0])), # input layer
    tf.keras.layers.Dense(300,activation=tf.nn.relu), # Dense or fully connected layer
    tf.keras.layers.Dense(len(np.unique(yTrain)),activation=tf.nn.softmax) # output layer with 10
])
model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model
model.fit(xTrain,yTrain, epochs=10)

loss, acc = model.evaluate(x=xTest,y=yTest)

print("loss=", loss, "acc=", acc)


# In[ ]:




