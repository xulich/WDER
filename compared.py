import pandas as pd
import numpy as np

from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import  KFold

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r'C:\Users\Administrator\Desktop\胃癌生存时间预测\dataset\pol.csv')
#data = pd.read_csv(r'C:\Users\Administrator\Desktop\胃癌生存时间预测\dataset\winequality-red.csv', delimiter=";")
#data=data[data['Survival months']<=60]
y = data['27']
X = data.drop('27',axis=1)


#model = GradientBoostingRegressor()
#model =  xgb.XGBRFRegressor()
#model = RandomForestRegressor()
#model = DecisionTreeRegressor()
#model = Ridge()
model = ElasticNetCV()
#model = Lasso()

kfold= KFold(n_splits=10, shuffle=True, random_state=2022)
rmses=[]
maes=[]
rs = []


for k,(train,test) in enumerate(kfold.split(X)):
    model.fit(X.iloc[train], y.iloc[train])
    y_pre = model.predict(X.iloc[test])
    rmse = np.sqrt(mean_squared_error(y.iloc[test], y_pre))
    mae = mean_absolute_error(y.iloc[test], y_pre)
    r = r2_score(y.iloc[test], y_pre)

    rmses.append(rmse)
    maes.append(mae)
    rs.append(r)
    result = pd.DataFrame(
        {
            "rmses": rmses,
            "maes": maes,
            "rs": rs
        }
    )


#print ('Fold:%s, rmse:%.6f, mae:%.6f, r:%.6f'%(k+1, rmse, mae, r))
print(result)

result.to_csv(r'C:\Users\Administrator\Desktop\胃癌生存时间预测\Results\pol\pol_EN.csv')
print(result.mean(axis=0))



