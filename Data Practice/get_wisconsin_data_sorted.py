import numpy as np
import pandas as pd

df = pd.read_csv('data.csv', delimiter = ',')
#delete whitespace surrounding column names
df.columns = df.columns.str.replace(' ', '')
#delete rows with missing bland chromatin
df = df[df.Bare_Nuclei != '?']


x = df.values
y = []
#create y using labels
x = list(x)
for row in x:
    row = list(row)
    y.append(row.pop())
x = np.array(x)

#delete label from x
x=x[:,:-1]
#delete id from x
x = x[:,1:]
print(x)
y = np.array(y)
print(y)
#notice we are not including id nor classification
feature_names = ['Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape', 'Marginal_Adhesion', 
'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']
#print(x.shape)
#print(y.shape)


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
#print(xTrain.shape)
#print(xTest.shape)
#print(yTrain.shape)
#print(yTest.shape)
