import os
import pandas as pd
import datetime
import numpy as np
import math
from sklearn import tree, preprocessing
from sklearn.cluster import KMeans
import pickle


# Reading data and adding test and train data together:
def get_train_data(data_set):
    df = pd.read_csv(data_set + '-training.csv', encoding="ISO-8859-1")
    df['event time:timestamp'] = df['event time:timestamp'].apply(to_date)
    return df


def get_test_data(data_set):
    df = pd.read_csv(data_set + '-test.csv', encoding="ISO-8859-1")
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


def remove_incomplete(data):
    #data.sort_values(by=['case concept:name', 'event time:timestamp'])
    #tails = data.groupby('case concept:name').tail(1)
    #heads = data.groupby('case concept:name').head(1)
    #begin_event = heads['event concept:name'].unique()[
    #    heads.groupby('event concept:name')['event concept:name'].transform('count').unique() >= 0.25 * len(heads)]
    #end_event = tails['event concept:name'].unique()[
    #    tails.groupby('event concept:name')['event concept:name'].transform('count').unique() >= 0.25 * len(tails)]
    #heads = heads[heads['event concept:name'].isin(begin_event)]
    #tails = tails[tails['event concept:name'].isin(end_event)]
    #data = data[data['case concept:name'].isin(heads['case concept:name']) & data['case concept:name'].isin(
    #    tails['case concept:name'])]
    return data


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


# Method for transforming events into horizontal format.
def transform_event_data(df):
    case = None
    data = {}
    for index, row in df.iterrows():
        if case != row['case concept:name']:
            case = row['case concept:name']
        if case not in data.keys():
            data[case] = []
        data[case].append(float(row['event concept:name']) + 1)
        data[case].append(float(row['inter-event-time']))
    df = pd.DataFrame.from_dict(data, orient='index')
    return df, len(df.columns)


def to_seconds(x):
    return x.total_seconds()


def get_blocked_means(nblocks, df_train):
    ordered_means = {}
    ordered_casedurations = df_train.groupby('case concept:name').head(1)[
        ['time-to-end', 'event time:timestamp']].sort_values('event time:timestamp')
    le = int(len(ordered_casedurations) / nblocks)
    k = 0
    for i in range(0, len(ordered_casedurations), le):
        if k == nblocks - 1:
            ordered_means[ordered_casedurations.iloc[i + le]['event time:timestamp']] = \
                ordered_casedurations.iloc[0:len(ordered_casedurations)]['time-to-end'].apply(to_seconds).mean()
            break
        else:
            ordered_means[ordered_casedurations.iloc[i + le]['event time:timestamp']] = \
                ordered_casedurations.iloc[0:i + le]['time-to-end'].apply(to_seconds).mean()
        k += 1
    return ordered_means


name = 'BPI_Challenge_2019'
nblocks = 100
df_train = remove_incomplete(get_train_data(name))
df_test = get_test_data(name)
print('got data')

df_train = get_time_to_end(df_train)
df_train = get_total_time_passed(df_train)
df_test = get_time_to_end(df_test)
df_test = get_total_time_passed(df_test)
print('got extra columns')

ordered_casedurations = df_train.groupby('case concept:name').head(1)[
    ['time-to-end', 'event time:timestamp']].sort_values('event time:timestamp')

ordered_means = get_blocked_means(nblocks, df_train)

estimated = []

for index, row in df_test.iterrows():
    for i in range(nblocks):
        if list(ordered_means.keys())[i] > row['event time:timestamp']:
            break
    if i == 0:
        estimated.append(-1)
    else:
        estimated.append(ordered_means[list(ordered_means.keys())[i - 1]])

df_test['estimator'] = estimated
df_test['estimator'] = df_test['estimator'] - df_test['inter-event-time'].apply(to_seconds)
df_test.loc[df_test['estimator'] < 0, 'estimator'] = 0
print(math.sqrt(sum((df_test['time-to-end'].apply(to_seconds) - df_test['estimator']).apply(lambda x: x ** 2)) / len(
    df_test)) / 3600 / 24)
df_train, le = convert_train(df_train)
df_test = convert_test(df_test, le)
df_test.to_pickle(name + 'predicted.dat')
df_train.to_pickle(name + 'extra-columns.dat')

#########

test_df = pd.read_pickle(name + 'predicted.dat')
train_df = pd.read_pickle(name + 'extra-columns.dat')

event_data_train, max_event = transform_event_data(train_df)
event_data_test, amount = transform_event_data(test_df)

# Calculating means per event type per cluster for train set:
clustering = KMeans(random_state=0, n_clusters=100)
event_data_train['cluster'] = clustering.fit_predict(event_data_train.fillna(0))
train_df.index = train_df['case concept:name']
train_df['cluster'] = event_data_train['cluster']
train_means = pd.DataFrame(train_df.groupby('case concept:name')['time-to-end'].max())
train_means['cluster'] = event_data_train['cluster']

# Making the test set of the right event length:
for i in range(amount, amount + (max_event - amount), 1):
    event_data_test[i] = np.nan

# Predict cluster on test set and use means per event type and per cluster as estimator:
event_data_test['cluster'] = clustering.predict(event_data_test.fillna(0))
test_df.index = test_df['case concept:name']
test_df['cluster'] = event_data_test['cluster']
for cluster in list(train_means.groupby('cluster').mean().index):
    test_df.loc[test_df['cluster'] == cluster, 'estimator2'] = float(train_means.groupby('cluster').mean().loc[cluster])
test_df['estimator2'] = test_df['estimator2'] - test_df['inter-event-time']
test_df.loc[test_df['estimator2'] < 0, 'estimator2'] = 0

# Calculate the error:
print(
    math.sqrt(sum((test_df['time-to-end'] - test_df['estimator2']).apply(lambda x: x ** 2)) / len(test_df)) / 3600 / 24)

# Save data again:
test_df.to_pickle(name + 'predicted.dat')
train_df.to_pickle(name + 'extra-columns.dat')
with open(name + 'names.dat', 'wb') as f:
    pickle.dump(le, f)
