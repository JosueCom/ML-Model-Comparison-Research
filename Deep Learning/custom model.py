from tensorflow.keras import Input, Dense, Flatten, LSTM
from keras.models import Model, Sequential
import pandas as pd 

input_size = 9

df = pd.read_csv("adjusted.csv")

data_input = Input(shape=(input_size,))

hidden = Dense(20, activation="softplus")(data_input)
hidden = Dense(10, activation="softplus")(hidden)
hidden = Dropout(0.8)(hidden)
data_output = Dense(2, activation="sigmoid")(hidden)

model = Model(inputs=data_input, outputs=data_output)

model.compile(metrics=["accuracy"])

model.fit()