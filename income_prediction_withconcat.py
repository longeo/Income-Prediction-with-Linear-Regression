# To do:
# 1. Find library which will pre-process data
# 2. Split training data to avoid overfitting 

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
import csv
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# IMPORT DATASET INFORMATION
dataset_train = pd.read_csv('/Users/eoinlong1/Desktop/College 19_20/Machine_Learning/tcd ml 2019-20 income prediction training (with labels).csv.zip')
dataset_test = pd.read_csv('/Users/eoinlong1/Desktop/College 19_20/Machine_Learning/tcd ml 2019-20 income prediction test (without labels).csv')

# CLEARN DATASET
dataset_train.isnull().any()
dataset_train = dataset_train.fillna(-1)
dataset_test.isnull().any()
dataset_test = dataset_test.fillna(-1)

# Remember which data belongs to training set so that concatenated database can be seperated later
train_index = len(dataset_train)
y_train = np.array(dataset_train['Income in EUR'])

# Drop income columns so that dataframe dimensions agree
dataset_train.drop('Income in EUR',axis='columns',inplace=True)
dataset_test.drop('Income',axis='columns',inplace=True)

# Create concatenated dataframe
# print(dataset_train.columns)
# print(dataset_test.columns)
df = pd.concat([dataset_train, dataset_test], axis=0)

df.drop('Instance',axis='columns',inplace=True)

# Deal with ordinal variables in data i.e University,
university_data = df[['University Degree']]
preserved_mapper = {'PhD':3, 'Master':2, 'Bachelor':1, -1:0, 'No':-1}
university_data = university_data.replace(preserved_mapper)
df.drop('University Degree',axis='columns',inplace=True)
df = pd.concat([df, university_data], axis=1)

# gender_data = df[['Gender']]
# preserved_mapper = {'male':1, 'female':-1, 'other':0, 'unknown':0, -1:0}
# gender_data = gender_data.replace(preserved_mapper)
# df.drop('Gender',axis='columns',inplace=True)
# df = pd.concat([df, gender_data], axis=1)

# Deal with categorical variables by creating dummy variables
df = pd.concat([df, pd.get_dummies(df['Gender'],prefix='Gender',prefix_sep=':')], axis=1)
df.drop('Gender',axis='columns',inplace=True)
df.drop('Gender:-1',axis='columns',inplace=True)
df.drop('Gender:unknown',axis='columns',inplace=True)

df = pd.concat([df, pd.get_dummies(df['Country'],prefix='Country',prefix_sep=':')], axis=1)
df.drop('Country',axis='columns',inplace=True)

df = pd.concat([df, pd.get_dummies(df['Hair Color'],prefix='Hair Color',prefix_sep=':')], axis=1)
df.drop('Hair Color',axis='columns',inplace=True)

df = pd.concat([df, pd.get_dummies(df['Profession'],prefix='Profession',prefix_sep=':')], axis=1)
df.drop('Profession',axis='columns',inplace=True)

df = pd.concat([df, pd.get_dummies(df['Wears Glasses'],prefix='Wears Glasses',prefix_sep=':')], axis=1)
df.drop('Wears Glasses',axis='columns',inplace=True)

# Split data back into test and training data
X_train = np.array(df.iloc[0:train_index])
X_test = np.array(df.tail(len(df) - (train_index)))

###### Random Forest Regressor (Method 2)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,oob_score=False, random_state=None, verbose=0, warm_start=False)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# ##### Random Forest Regressor with tuned Hyper parameters (Method 3)
# A hyperparameter is a parameter whose value is set before the learning process begins.
# from sklearn.model_selection import GridSearchCV
# param_grid = [
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  # ]
# forest_reg = RandomForestRegressor()
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)
# y_pred = grid_search.predict(X_test)

# Write results to .csv file
with open('y_pred.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(map(lambda x: [x],y_pred))
csvFile.close()







