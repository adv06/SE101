import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


DATA_LENGTH = 1000

#loaded dataset from csv file
df = pd.read_csv('tim_hortons_weather_time_people_dataset.csv')

#set encoder
encoder = OneHotEncoder(sparse=False)
weather_encoded = encoder.fit_transform(df['Weather'].values.reshape(-1, 1))

# Step 2: Time of Day 
time_data = df['Time of Day (minutes)'].values
time_data = time_data / (7 * 24 * 60)  # Normalize the time between 0 and 1

# Step 3: Number of People 
people_data = df['Number of People'].values

# Step 4: Create input tensors for TensorFlow
x1_train = tf.constant(weather_encoded, dtype=tf.float32)  # One-hot encoded weather data
x2_train = tf.constant(time_data.reshape(-1, 1), dtype=tf.float32)  # Normalized time data
y_train = tf.constant(people_data, dtype=tf.float32)  # Number of people


# # weather

# x1_train = tf.constant([])

# # time of week (minutes)
# x2_train = tf.constant([])

# #of people at tim hortons
# y_train = tf.constant([])

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

