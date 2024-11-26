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
weather_encoded = np.array([int(np.where(weather_encoded[i] == 1)[0]) for i in range(len(weather_encoded))])

# Step 2: Time of Day 
time_data = df['Time of Day (minutes)'].values
time_data = time_data / (7 * 24 * 60)  # Normalize the time between 0 and 1

# Step 3: Number of People 
people_data = df['Number of People'].values
scale = max(people_data)/30
print(max(people_data), scale)
people_data = people_data/scale

# print(weather_encoded)

# print(time_data)
print(people_data)