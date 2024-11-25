import tensorflow as tf
from tensorflow import keras
import numpy as np

DATA_LENGTH = 1000
# weather
x1_train = tf.constant([])

# time of week (minutes)
x2_train = tf.constant([])

#of people at tim hortons
y_train = tf.constant([])

model = keras.Sequential([
    # First hidden
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(10, activation = "relu"),
    keras.layers.Dense(5, activation = "relu")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# epochs - same image in different order
model.fit(x1_train, y_train, epochs=10)

model2 = keras.Sequential([
    # First hidden
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(10, activation = "relu"),
    keras.layers.Dense(5, activation = "relu"),
])

model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# epochs - same image in different order
model2.fit([x1_train, x2_train], y_train, epochs=10)

model3 = tf.concat([model.output, model2.output])


fnal = keras.layers.Dense(2, activation="relu")(model3)
fnal = keras.layers.Dense(2, activation="linear")(fnal)


loss, acc = fnal.evaluate([x1_train, x2_train], y_train)
print("Accuracy: ",  acc)

