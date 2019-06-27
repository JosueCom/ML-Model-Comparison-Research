from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Precision
import pandas as pd 
import numpy as np

input_size = 9

df = pd.read_csv("../Data Practice/adjusted.csv")

X = df.iloc[:, 1:-1].to_numpy()
Y = df.iloc[:, -1].to_numpy()

data_input = Input(shape=(input_size,))

hidden = Dense(20, activation="softplus")(data_input)
hidden = Dense(10, activation="softmax")(hidden)
hidden = Dropout(0.2)(hidden)
data_output = Dense(1, activation="sigmoid")(hidden)

model = Model(inputs=data_input, outputs=data_output)

# compile model
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[Precision(), "accuracy"])

model.fit(X, Y, epochs=50)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))

model.summary()

# model.save("custom_model.h5")