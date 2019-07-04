#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sb
df = pd.read_csv("adjusted.csv")
X = np.array(df.drop(columns=[' Class']))
Y = np.array(df.iloc[:, -1])
random_state = 0
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state=random_state)
yTrain, yTest = yTrain[:,np.newaxis], yTest[:,np.newaxis]


# In[3]:


model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=np.shape(xTrain[0])),
    tf.keras.layers.Dense(150, activation=tf.nn.relu),
    tf.keras.layers.Dense(110, activation=tf.nn.relu),
    tf.keras.layers.Dense(len(np.unique(yTrain)), activation=tf.nn.softmax)
])
model1.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(xTrain, yTrain, epochs=100)
loss, acc = model1.evaluate(x=xTest, y=yTest)
print('loss=', loss, 'acc=', acc)


# In[139]:


guesses0 = model0.predict(xTest)
guesses1 = model1.predict(xTest)
guesses_total = []


for i in range(len(xTest)):
    benign = np.asscalar(guesses0[i][0]+guesses1[i][0])
    malignant = np.asscalar(guesses0[i][1]+guesses1[i][1])
    if benign > malignant:
        guesses_total.append(0)
    else:
        guesses_total.append(1)
print(len(guesses_total))


# In[91]:


print(len(xTrain))


# In[92]:


from sklearn.metrics import accuracy_score
print(accuracy_score(guesses_total, yTest))


# In[ ]:




