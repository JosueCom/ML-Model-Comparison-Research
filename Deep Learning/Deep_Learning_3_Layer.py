#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sb


# In[2]:


df = pd.read_csv("adjusted.csv")
X = np.array(df.drop(columns=[' Class']))
Y = np.array(df.iloc[:, -1])


# In[3]:


xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state=3)
yTrain, yTest = yTrain[:,np.newaxis], yTest[:,np.newaxis]


# In[4]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=np.shape(xTrain[0])),
    tf.keras.layers.Dense(300, activation=tf.nn.relu),
    tf.keras.layers.Dense(len(np.unique(yTrain)), activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain, epochs=10)
loss, acc = model.evaluate(x=xTest, y=yTest)
print('loss=', loss, 'acc=', acc)


# In[ ]:




