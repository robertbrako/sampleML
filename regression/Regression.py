# coding: utf-8

from pandas import Series, DataFrame
import pandas as pandz
import numpy as numpz
import numpy.random as numrand
import random
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import sys

sys.setrecursionlimit(10000)

class Regression:
    def __init__(self, csv_file = None, data = None, values = None):
        if (data is None and csv_file is not None):
            dataFile = pandz.read_csv(csv_file, nrows=1000);
            self.values = dataFile['AppraisedValue']
            
            self.dataFile = dataFile;
            self.dataFile = self.dataFile[['lat', 'long', 'SqFtLot']]
        elif (data is not None and values is not None):
            self.dataFile = data
            self.values = values
        else:
            raise ValueError("Must have either csv_file or data set")
            
        self.n = len(self.dataFile)
        self.kdtree = KDTree(self.dataFile)
        self.metric = numpz.mean
        self.k = 5

def regress(self, query_point):
    distances, indexes = self.kdtree.query(query_point, self.k)
    m = self.metric(self.values.iloc[indexes])
    if numpz.isnan(m):
        raise Exception('Unexpected result')
    else:
        return m

def error_rate(self, folds):
    holdout = 1 / float(folds)
    errors = []
    for fold in range(folds):
        y_hat, y_true = __validation_data(self, holdout)
        errors.append(mean_absolute_error(y_true, y_hat))
    return errors

def __validation_data(self, holdout):
    test_rows = random.sample(self.dataFile.index.tolist(), int(round(len(self.dataFile) * holdout)))
    train_rows = set(range(len(self.dataFile))) - set(test_rows)
    df_test = self.dataFile.loc[test_rows]
    df_train = self.dataFile.drop(test_rows)
    test_values = self.values.loc[test_rows]
    train_values = self.values.loc[train_rows]
    kd = Regression(data = df_train, values = train_values)
    
    y_hat = []
    y_actual = []
    
    for idx, row in df_test.iterrows():
        y_hat.append(regress(kd, row))
        y_actual.append(self.values[idx])
    return (y_hat, y_actual)

def plot_error_rates(self):
    folds = range(2, 11)
    errors_df = pandz.DataFrame({'max':0, 'min':0}, index=folds)
    for fold in folds:
        error_rates = error_rate(self, fold)
        errors_df['max'][fold] = max(error_rates)
        errors_df['min'][fold] = min(error_rates)
    errors_df.plot(title='MAE of KNN over different folds')
    plt.show()

regz = Regression('king_county_data_geocoded.csv', None, None)
regress(regz, [47.4,-122,1600])

"3 Folds means 2/3 of data is for training, 1/3 of data is for testing"
error_rate(regz, 3)
plot_error_rates(regz)
