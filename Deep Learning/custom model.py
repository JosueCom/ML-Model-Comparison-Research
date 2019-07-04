from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Precision, Recall
from keras import backend as K
import pandas as pd 
import numpy as np



def specificity(y_pred, y_true):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    Specificity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity



input_size = 9


df = pd.read_csv("../Data Practice/adjusted.csv")

X = df.iloc[:, 1:-1].to_numpy()
Y = df.iloc[:, -1].to_numpy()


n_train = int(X.shape[0] * 0.9)

data_input = Input(shape=(input_size,))

hidden = Dense(20, activation="softplus")(data_input)
hidden = Dense(10, activation="softmax")(hidden)
hidden = Dropout(0.2)(hidden)
data_output = Dense(1, activation="sigmoid")(hidden)

model = Model(inputs=data_input, outputs=data_output)

# compile model
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy", Precision(), Recall(), specificity])

model.fit(X[:n_train], Y[:n_train], epochs=100)

scores = model.evaluate(X[n_train:], Y[n_train:])
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
print("\n%s: %.2f%%" % (model.metrics_names[3], scores[3]*100))
print("\n%s: %.2f%%" % (model.metrics_names[4], scores[4]*100))

model.summary()

model.save("custom_model.h5")