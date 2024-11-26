import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

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
scale = max(people_data)/30  
people_data = people_data/scale

# Step 4: Create input tensors for TensorFlow
x1_train = tf.constant(weather_encoded, dtype=tf.float32)  # One-hot encoded weather data
x2_train = tf.constant(time_data.reshape(-1, 1), dtype=tf.float32)  # Normalized time data
y_train = tf.constant(people_data, dtype=tf.float32)/30  # Number of people



# Create a line plot
plt.scatter(x2_train, y_train, label='bruh', color='blue', linewidth=2)
fig = plt.gcf()
fig.set_size_inches(12, 8)  # Width of 12 inches and height of 8 inches
# Add title and labels
plt.title('Line Plot Example')
plt.xlabel('X values')
plt.ylabel('Y values')

# Show legend
plt.legend()

# Display the plot
plt.show()
