import os
import pandas as pd
import datetime
import numpy as np
import math


# Reading data and adding test and train data together:
def get_train_data(data_set):
    df = pd.read_csv(data_set + '-training.csv')
    df['event time:timestamp'] = df['event time:timestamp'].apply(to_date)
    return df


def get_test_data(data_set):
    df = pd.read_csv(data_set + '-test.csv')
    df['event time:timestamp'] = df['event time:timestamp'].apply(to_date)
    return df


# Convert the time column to a python datetime object:
def to_date(x):
    # Date in format 'day-month-year hour:min:sec.milsec':
    day, month, year = str(x).split(" ")[0].split("-")
    hour, minute, sec = str(x).split(" ")[1].split(":")
    second, milsec = sec.split(".")
    return datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), int(milsec) * 1000)


def to_seconds(x):
    return x.total_seconds()


def get_time_to_end(df):
    df = df.sort_values(by=['case concept:name', 'event time:timestamp'], ascending=[True, False])
    case = None
    inter_event_times = []
    for index, row in df.iterrows():
        current = row['case concept:name']
        if case != current:
            case = current
            start = row['event time:timestamp']
            inter_event_times.append(datetime.timedelta(0))
        else:
            end = row['event time:timestamp']
            inter_event_times.append(start - end)
    df['time-to-end'] = inter_event_times
    return df


def get_total_time_passed(df):
    df = df.sort_values(by=['case concept:name', 'event time:timestamp'])
    case = None
    inter_event_times = []
    total_times = []
    end = None
    for index, row in df.iterrows():
        current = row['case concept:name']
        if case != current:
            case = current
            start = row['event time:timestamp']
            inter_event_times.append(datetime.timedelta(0))
        else:
            end = row['event time:timestamp']
            inter_event_times.append(end - start)
    df['inter-event-time'] = inter_event_times
    return df


def train_naive_estimator(df):
    total_times = pd.DataFrame(df.groupby(by='case concept:name')['time-to-end'].max()).reset_index()
    average = total_times['time-to-end'].mean()
    return average


def predict_naive(average, df):
    df['estimator'] = average - df['inter-event-time']
    return df


def calculate_error(df):
    return math.sqrt(
        sum((df['time-to-end'] - df['estimator']).apply(to_seconds).apply(lambda x: x ** 2)) / len(df)) / 3600 / 24


name = 'BPI_Challenge_2012'
df_train = get_train_data(name)
df_test = get_test_data(name)
print('got data')

df_train = get_time_to_end(df_train)
df_train = get_total_time_passed(df_train)
df_test = get_time_to_end(df_test)
df_test = get_total_time_passed(df_test)
print('got extra columns')

mean = train_naive_estimator(df_train)
df_test = predict_naive(mean, df_test)
error = calculate_error(df_test)
print(error)

df_test.to_csv(name + 'predicted.csv')
df_train.to_csv(name + 'extra-columns.csv')
