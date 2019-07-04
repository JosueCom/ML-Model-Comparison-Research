from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.metrics import Precision
import pandas as pd 
import numpy as np

df = pd.read_csv("../Data Practice/adjusted.csv")

X = df.iloc[:, 1:-1].to_numpy()
Y = df.iloc[:, -1].to_numpy()


model = load_model("custom_model.h5")

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
print("\n%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
print("\n%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))

model.summary()

# model.save("custom_model.h5")