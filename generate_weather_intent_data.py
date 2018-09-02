# script used to generate synthetic training data for weather queries
import pandas as pd
import numpy as np
import csv

cities = pd.read_csv('data/cities.csv')
countries = ['Canada', 'Japan', 'United States', 'France', 'United Kingdom', 'Spain',
             'Germany', 'China']
cities = [cities[cities['country'] == country]['name'].tolist() for country in countries]
cities = [item for sublist in cities for item in sublist]
cities = [city for city in cities if city.isalpha()]
cities = list(map(lambda x: x.lower(), cities))
cities += ['singapore', 'bangkok', 'bali', 'phuket']

with open('data/weather_questions.txt') as f:
    text = f.readlines()

weather_queries = list(filter(lambda x: len(x) > 0, ["BOS " + x.lower().strip() + " EOS" for x in text]))
weather_vocab = []

weather_forecast_temperature = ['weather', "temperature", "forecast"]

time_horizon = ["this morning", "this month", "next month", "tonight", "this evening", "this afternoon", "tomorrow",
                "this week", "next week",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today"]

weather_conditions = ["cold", "chilly", "sunny", "sun",
                      "fog", "foggy", "stormy", "warm", "breezy",
                      "raining", "snowing", "rain", "snow", "rainy",
                      "snowy", "cloudy", "dry", "windy", "wind", "tornado",
                      "rains", "snows", "storms", 'thunderstorm', 'nice', 'good', 'bad']

weather_query_slots = ['O', 'B-city_name', 'I-city_name', 'B-weather_forecast_temperature',
                       'B-weather_condition', 'B-time_horizon', 'I-time_horizon']


def get_weather_query_slots(sentence):
    sentence = sentence.split(" ")
    slots = []
    i = 0
    while i < len(sentence):
        if ' '.join(sentence[i:i + 3]) in cities and len(sentence[i:i + 3]) == 3:
            slots.append('B-city_name')
            slots.append('I-city_name')
            slots.append('I-city_name')
            i += 3
        elif ' '.join(sentence[i:i + 2]) in cities and len(sentence[i:i + 2]) == 2:
            slots.append('B-city_name')
            slots.append('I-city_name')
            i += 2
        elif sentence[i] in cities:
            slots.append('B-city_name')
            i += 1
        elif sentence[i] in weather_forecast_temperature:
            slots.append('B-weather_forecast_temperature')
            i += 1
        elif sentence[i] in weather_conditions:
            slots.append('B-weather_condition')
            i += 1
        elif ' '.join(sentence[i:i + 2]) in time_horizon and len(sentence[i:i + 2]) == 2:
            slots.append('B-time_horizon')
            slots.append('I-time_horizon')
            i += 2
        elif sentence[i] in time_horizon:
            slots.append('B-time_horizon')
            i += 1
        else:
            slots.append('O')
            i += 1
    return slots


def generate_weather_queries(stop=3000, weather_vocab=weather_vocab):
    queries = []
    queries_slots = []
    np.random.seed(1000)
    while len(queries) < stop:
        i = np.random.randint(0, len(weather_queries))
        j = np.random.randint(0, len(cities))
        k = np.random.randint(0, len(weather_forecast_temperature))
        l = np.random.randint(0, len(weather_conditions))
        m = np.random.randint(0, len(time_horizon))
        new_city = cities[j]
        new_time_horizon = time_horizon[m]
        new_wft = weather_forecast_temperature[k]
        new_weather_condition = weather_conditions[l]
        query = weather_queries[i]
        query_list = query.split(' ')
        slots = get_weather_query_slots(query)
        city_substrings = [query_list[i] for i in range(len(slots)) if slots[i] in ['B-city_name', 'I-city_name']]
        city = " ".join(city_substrings)
        query = query.replace(city, new_city)
        horizon_substrings = [query_list[i] for i in range(len(slots)) if
                              slots[i] in ['B-time_horizon', 'I-time_horizon']]
        if horizon_substrings:
            horizon = " ".join(horizon_substrings)
            query = query.replace(horizon, new_time_horizon)
        wft = [query_list[i] for i in range(len(slots)) if slots[i] is 'B-weather_forecast_temperature']
        if wft:
            query = query.replace(wft, new_wft)
        weather_condition = [query_list[i] for i in range(len(slots)) if slots[i] is 'B-weather_condition']
        if weather_condition:
            query = query.replace(weather_condition, new_weather_condition)
        if len(query) < 150:
            l = query.split(' ')
            for word in l:
                weather_vocab.append(word)
            queries.append(l)
            queries_slots.append(get_weather_query_slots(query))

        with open('data/weather.dict.vocab.csv', 'w', newline='') as myfile:
            writer = csv.writer(myfile, delimiter='\n')
            text = sorted(list(set(weather_vocab)))
            writer.writerow(text)

    return queries, queries_slots
