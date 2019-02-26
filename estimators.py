import os
import pandas as pd
import datetime
import numpy as np
import math
from sklearn.cluster import KMeans

# Method for transforming events into horizontal format.
def transform_event_data(df):
    case = None
    data = {}
    for index, row in df.iterrows():
        if case != row['case concept:name']:
            case = row['case concept:name']
        if case not in data.keys():
            data[case] = []
        data[case].append(float(row['event concept:name']))
    df = pd.DataFrame.from_dict(data, orient='index')
    return df, len(df.columns)


def to_seconds(x):
    return x.total_seconds()


name = 'Road_Traffic_Fine_Management_Process'
test_df = pd.read_pickle(name + 'predicted.dat')
train_df = pd.read_pickle(name + 'extra-columns.dat')

event_data_train, max_event = transform_event_data(train_df)
event_data_test, amount = transform_event_data(test_df)

# Calculating means per event type per cluster for train set:
clustering = KMeans(random_state=0)
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

# Calculate the error:
print(
    math.sqrt(sum((test_df['time-to-end'] - test_df['estimator2']).apply(lambda x: x ** 2)) / len(test_df)) / 3600 / 24)

# Save data again:
test_df.to_pickle(name + 'predicted.dat')
train_df.to_pickle(name + 'extra-columns.dat')
