import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


DATA_LENGTH = 1000

#loaded dataset from csv file
df = pd.read_csv("C:/Advey/SE101Clone/SE101/tim_hortons_weather_time_people_dataset.csv")

#set encoder
encoder = OneHotEncoder()
weather_encoded = arr=encoder.fit_transform(df['Weather'].values.reshape(-1, 1)).toarray()
# weather_encoded = np.array([int(np.where(weather_encoded[i] == 1)[0]) for i in range(len(weather_encoded))])

# Step 2: Time of Day 
time_data = df['Time of Day (minutes)'].values
time_data = time_data / (7 * 24 * 60)  # Normalize the time between 0 and 1

# Step 3: Number of People 
people_data = df['Number of People'].values
scale = max(people_data)/30
people_data = people_data/scale

# Step 4: Create input tensors for TensorFlow
x1_train = tf.constant(weather_encoded, dtype=tf.float32)  # One-hot encoded weather data
x2_train = tf.constant(time_data.reshape(-1, 1), dtype=tf.float32)  # Normalized time data
y_train = tf.constant(people_data, dtype=tf.float32)  # Number of people

print(f"x2_train shape: {x2_train.shape}")  # Should print (batch_size, 1)
print(f"y_train shape: {y_train.shape}")    # Should print (batch_size,)

# functional models
input1 = keras.layers.Input(shape=(4,))
x1 = keras.layers.Dense(200, activation="relu") (input1)
x2 = keras.layers.Dense(100, activation = "relu") (x1)
x1 = keras.layers.Dense(10, activation = "relu") (x1)
x1 = keras.layers.Dense(5, activation = "relu") (x1)
x1 = keras.Model(inputs=input1, outputs=x1)

input2 = keras.layers.Input(shape=(1,))
x2 = keras.layers.Dense(200, activation="relu") (input1)
x2 = keras.layers.Dense(100, activation = "relu") (x2)
x2 = keras.layers.Dense(10, activation = "relu") (x2)
x2 = keras.layers.Dense(5, activation = "relu") (x2)
x2 = keras.Model(inputs=input2, outputs=x2)

combine = keras.layers.concatenate([x1.output, x2.output])
x3 = keras.layers.Dense(2, activation="relu") (combine)
x3 = keras.layers.Dense(1, activation="linear") (x3)
model = keras.Model(inputs=[x1.input, x2.input], outputs=x3)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
model.fit([x1_train, x2_train], y_train, epochs=10) 
loss, mae = model.evaluate([x1_train, x2_train], y_train) 

print("Mean Absolute Error: ", mae)
predicted_value = model.predict([np.array([1, 0, 0, 0]).reshape(1, -1), np.array([40]).reshape(1, -1)])
print(predicted_value)


