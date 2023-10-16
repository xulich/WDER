import pandas as pd
import numpy as np

from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from DER import DER
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression


data = pd.read_csv(r'C:\Users\Administrator\Desktop\胃癌生存时间预测\dataset\abalone.csv')
#data = pd.read_csv(r'C:\Users\Administrator\Desktop\胃癌生存时间预测\dataset\winequality-red.csv', delimiter=";")
#data=data[data['Survival months']<=60]
y = data['Age']
X = data.drop('Age',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


results = pd.DataFrame(
    {
        "n_clusters":[],
        "beta":[],
        "eta":[],
        "alpha":[],
        "rmses": [],
        "maes": [],
        "rs": []
    })
max_iterations = 500
for n_clusters in [3, 5]:
    for beta in [0, 2, 4, 6, 8, 10, 12, 14]:
        for eta in [0.7, 0.8, 0.9, 1]:
            for alpha in [0.2, 0.3, 0.35, 0.4, 0.5]:
                y_pre = DER(X_train, y_train, X_test, n_clusters, beta, max_iterations, eta, alpha,
                            model_name='GBRT')
                rmse = np.sqrt(mean_squared_error(y_test, y_pre))
                mae = mean_absolute_error(y_test, y_pre)
                r = r2_score(y_test, y_pre)

                result = pd.DataFrame(
                    {
                        "n_clusters": [n_clusters],
                        "beta": [beta],
                        "eta": [eta],
                        "alpha": [alpha],
                        "rmses": [rmse],
                        "maes": [mae],
                        "rs": [r]
                    })

                results = results.append(result, ignore_index=True)
print(results)
results.to_csv(r'C:\Users\Administrator\Desktop\胃癌生存时间预测\Results\abalone_search.csv')
