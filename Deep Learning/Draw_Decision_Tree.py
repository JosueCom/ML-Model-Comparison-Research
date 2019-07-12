#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
df = pd.read_csv("adjusted.csv")
print(df.columns)
X = np.array(df.drop(columns=[' Class']))
Y = np.array(df.iloc[:, -1])
random_state = 0
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state=random_state)
yTrain, yTest = yTrain[:,np.newaxis], yTest[:,np.newaxis]
dt = tree.DecisionTreeClassifier()
dt.fit(xTrain, yTrain)

dotfile = open("dt.dot", 'w')
tree.export_graphviz(dt, out_file=dotfile, feature_names=['Clump_Thickness', 
                                                          'Uniformity_of_Cell_Size',
                                                          'Uniformity_of_Cell_Shape',
                                                          'Marginal_Adhesion',
                                                          'Single_Epithelial_Cell_Size', 
                                                          'Bare_Nuclei', 
                                                          'Bland_Chromatin',
                                                          'Normal_Nucleoli',
                                                          'Mitoses',])
dotfile.close()


# In[2]:


import pydot

(graph,) = pydot.graph_from_dot_file('dt.dot')
graph.write_png('dt.png')


# In[3]:





# In[ ]:




