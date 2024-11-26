import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def convert_degree(X, deg):
    # add the polynomial with different degress
    newarr = X
    for i in range(1, deg+1):
        newarr = np.hstack((newarr, X**i))
    return newarr 

def predict(time_of_day):
    predicted_value = 1+model.predict([convert_degree(np.array([time_of_day/(7 * 24 * 60)]), 14).reshape(1, -1)])*35
    return predicted_value



#loaded dataset from csv file
df = pd.read_csv("C:/Advey/SE101Clone/SE101/tim_hortons_data.csv")

#set encoder
encoder = OneHotEncoder()
weather_encoded = arr=encoder.fit_transform(df['weather'].values.reshape(-1, 1)).toarray()
# weather_encoded = np.array([int(np.where(weather_encoded[i] == 1)[0]) for i in range(len(weather_encoded))])

# Step 2: Time of Day 
time_data = df['time_of_day'].values
time_data = time_data / (7 * 24 * 60)  # Normalize the time between 0 and 1

# Step 3: Number of People 
people_data = df['num_people'].values
scale = max(people_data) 
people_data = people_data / scale

# Step 4: Create input tensors for TensorFlow
# x1_train = tf.constant(weather_encoded, dtype=tf.float32)  # One-hot encoded weather data
x2_train = tf.constant(convert_degree(time_data.reshape(-1, 1), 14), dtype=tf.float32)  # Normalized time data
y_train = tf.constant(people_data, dtype=tf.float32)  # Number of people

print(f"x2_train shape: {x2_train.shape}")  # Should print (batch_size, 1)
print(f"y_train shape: {y_train.shape}")    # Should print (batch_size,)

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(15,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="linear")
])
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
model.fit(x2_train, y_train, epochs=50, batch_size=32) 


print(predict(1080))
