#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sb


# In[12]:


df = pd.read_csv("adjusted.csv")
X = np.array(df.drop(columns=[' Class']))
Y = np.array(df.iloc[:, -1])
random_state = 0


# In[13]:


xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state=random_state)
yTrain, yTest = yTrain[:,np.newaxis], yTest[:,np.newaxis]


# In[14]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=np.shape(xTrain[0])),
#in order to change (for example) from 200 to 300 neurons, simply change the following number
    tf.keras.layers.Dense(80, activation=tf.nn.relu),
    tf.keras.layers.Dense(len(np.unique(yTrain)), activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#EPOCHS NEEDS TO BE AT LEAST 100
model.fit(xTrain, yTrain, epochs=100)
loss, acc = model.evaluate(x=xTest, y=yTest)
print('loss=', loss, 'acc=', acc)


# In[8]:


guesses = model.predict(xTest)


# In[9]:


len(yTrain[0])


# In[10]:


guesses[0]


# In[11]:


#add up all the guesses for various models and see which is biggest


# In[ ]:




