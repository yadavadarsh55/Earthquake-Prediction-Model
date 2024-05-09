import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Earthquake.csv")
dataset.drop(columns=['dmin','net', 'id', 'updated', 'place', 'type',
                      'horizontalError', 'depthError', 'magError', 'magNst', 'status',
                      'locationSource', 'magSource','rms'], axis = 1, inplace=True )

dataset['time'] = pd.to_datetime(dataset['time'])
dataset['year'] = dataset['time'].dt.year
dataset['month'] = dataset['time'].dt.month_name()
dataset['day'] = dataset['time'].dt.day_name()
dataset['day_of_week'] = dataset['time'].dt.dayofweek
def map_month_to_season(month):
    if month in ['March', 'April', 'May']:
        return 'Spring'
    elif month in ['June', 'July', 'August']:
        return 'Summer'
    elif month in ['September', 'October', 'November']:
        return 'Fall'
    else:
        return 'Winter'
dataset['season'] = dataset['month'].apply(map_month_to_season)
dataset['hour'] = dataset['time'].dt.hour
def hour_to_tod(hour):
    if hour in [4,5,6,7,8,9]:
        return 'Morning'
    elif hour in [10,11,12,13,14,15,16]:
        return 'Noon'
    elif hour in [17,18,19]:
        return 'Evening'
    else:
        return 'Night'
dataset['time_of_day'] = dataset['hour'].apply(hour_to_tod)

dataset = dataset.drop(columns=['time','month','day','hour'])
mag = dataset.pop('mag').values
magType = dataset.pop('magType').values
dataset.insert(9, value=mag, column='mag')
dataset.insert(10, value=magType, column='magType')

X = dataset.iloc[:,:-2].values
y_mag = dataset.iloc[:,-2].values
y_magType = dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[: ,:-2])
X[: , :-2] = imputer.transform(X[: ,:-2])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_mag, test_size=0.25, random_state=0)

from sklearn.preprocessing import LabelEncoder
le_train = LabelEncoder()
le_test = LabelEncoder()
X_train[:,-2] = le_train.fit_transform(X_train[:,-2])
X_test[:,-2] = le_test.fit_transform(X_test[:,-2])
X_train[:,-1] = le_train.fit_transform(X_train[:,-1])
X_test[:,-1] = le_test.fit_transform(X_test[:,-1])

from sklearn.ensemble import RandomForestRegressor
regressor3 = RandomForestRegressor(n_estimators=500, random_state=0)
regressor3.fit(X_train,y_train)

y_pred3 = regressor3.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred3))

from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator= regressor3, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuricies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuricies.std()*100))
