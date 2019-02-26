import os
import pandas as pd
import datetime
import numpy as np
import math
from sklearn import tree, preprocessing


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
    df['estimator'] = df['estimator'].apply(to_seconds)
    return df


def convert_train(df):
    le = preprocessing.LabelEncoder()
    le.fit(df['event concept:name'])
    df['event concept:name'] = le.transform(df['event concept:name'])
    df['inter-event-time'] = df['inter-event-time'].apply(to_seconds)
    df['time-to-end'] = df['time-to-end'].apply(to_seconds)
    return df, le


def convert_test(df, le):
    df['inter-event-time'] = df['inter-event-time'].apply(to_seconds)
    df['event concept:name'] = le.transform(df['event concept:name'])
    df['time-to-end'] = df['time-to-end'].apply(to_seconds)
    return df


def train_basic_tree(df):
    X = df[['inter-event-time', 'event concept:name']]
    Y = df['time-to-end']
    clf = tree.DecisionTreeRegressor(max_depth=5)
    clf = clf.fit(X, Y)
    return clf, le


def predict_basic_tree(df, clf, le):
    df['estimator 2'] = clf.predict(df[['inter-event-time', 'event concept:name']])
    return df


def calculate_error(df):
    df['error'] = df['time-to-end'] - df['estimator']
    return df


def calculate_error2(df):
    return math.sqrt(
        sum((df['time-to-end'].apply(to_seconds) - df['estimator 2']).apply(lambda x: x ** 2)) / len(df)) / 3600 / 24


print('input file name')
name = input()
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
df_train, le = convert_train(df_train)
df_test = convert_test(df_test, le)
df_test = calculate_error(df_test)

df_test.to_pickle(name + 'predicted.dat')
df_train.to_pickle(name + 'extra-columns.dat')
