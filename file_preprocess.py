import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing

train = pd.read_csv('exposition_train.tsv', sep='\t', low_memory=False, header=0)
target = train['target']
id = train['id']
features = train.drop(['area', 'building_series_id', 'site_id', 'target_string', 'main_image', 'latitude', 'building_id', 'unified_address', 'day', 'longitude', 'target', 'id', 'locality_name', 'parking'], 1)
features.insert(len(features.columns), 'parking', train['parking'])
to_normalize = [(0, 'build_year'), (3, 'ceiling_height'), (4, 'rooms'), (5, 'floors_total'), (6, 'living_area'), (7, 'floor'), (2, 'total_area'), (11, 'kitchen_area'), (12, 'price'), (13, 'flats_count'), (14, 'building_type'), (15, 'balcony'), (16, 'renovation'), (17, 'parking')]
for pos, labl in to_normalize:
  temp = list(features[labl].to_numpy())
  if pos == 0 or 14 <= pos:
    remont_encoder = preprocessing.LabelEncoder()
    remont_encoder.fit(list(temp))
    eh = list(enumerate(remont_encoder.classes_))
    for i in range(len(eh)):
      new_i = (eh[i][1], eh[i][0])
      eh[i] = new_i
    eh = dict(eh)
    for i in range(len(temp) - 1):
      temp[i] = float(eh[temp[i]])
  for i in range(len(temp)):
    if pos == 17 and temp[i] == 'UNKNOWN':
      temp[i] = 0
    temp[i] = float(temp[i])
  maximum = max(temp)
  for i in range(len(temp)):
    temp[i] /= maximum
  features = features.drop(labl, 1)
  features.insert(pos, labl, temp)

to_transform = [(1, 'expect_demolition'), (8, 'is_apartment'), (9, 'has_elevator'), (10, 'studio')]
for pos, labl in to_transform:
  temp = list(features[labl].to_numpy())
  for i in range(len(temp)):
    if temp[i] == False:
      temp[i] = float(0)
    else:
      temp[i] = float(1)
  features = features.drop(labl, 1)
  features.insert(pos, labl, temp)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state = 0) 
features.head()
