import random
import csv

def generate_weather(day_num):
    if day_num == 0:
        return "sunny"
    elif day_num == 1:
        return "rainy"
    elif day_num == 2:
        return "cloudy"
    else:
        return "snowy"

def generate_people(time_of_day):
    peak_morning = 9 * 60  # 9 AM
    peak_afternoon = 12 * 60  # 12 PM
    peak_evening = 19 * 60  # 7 PM

    time_diff_morning = abs(time_of_day - peak_morning)
    time_diff_afternoon = abs(time_of_day - peak_afternoon)
    time_diff_evening = abs(time_of_day - peak_evening)

    # Adjust the peak values and decay rates as needed to control the number of people
    people = 100 - min(time_diff_morning, time_diff_afternoon, time_diff_evening) * 2

    return max(0, people)

with open('tim_hortons_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['weather', 'time_of_day', 'num_people']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(3000):
        time_of_day = random.randint(0, 10080)
        day_num = time_of_day // (60 * 24)
        weather = generate_weather(random.randint(0, 3))
        num_people = generate_people(time_of_day-day_num*60*24)
        writer.writerow({'weather': weather, 'time_of_day': time_of_day, 'num_people': num_people})