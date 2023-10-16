import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  KFold
from sklearn.metrics import r2_score
from DER import DER

data = pd.read_csv(r'C:\Users\Administrator\Desktop\胃癌生存时间预测\dataset\pol.csv')
#data = pd.read_csv(r'C:\Users\Administrator\Desktop\胃癌生存时间预测\dataset\winequality-red.csv', delimiter=";")
#data=data[data['Survival months']<=60]
y = data['27']
X = data.drop('27',axis=1)

n_clusters = 5
beta = 0 # w的尺度变化系数
max_iterations = 500
eta = 0.9  #采样率
alpha = 0.35  #权重归一化系数

kfold= KFold(n_splits=10, shuffle=True, random_state=2022)
rmses=[]
maes=[]
rs = []

for k,(train,test) in enumerate(kfold.split(X)):
    y_pre = DER(X.iloc[train], y.iloc[train],X.iloc[test], n_clusters, beta, max_iterations, eta, alpha, model_name='RFR')
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

    print ('Fold:%s, rmse:%.6f, mae:%.6f, r:%.6f'%(k+1, rmse, mae, r))
print(result)
print(result.mean(axis=0))
result.to_csv(r'C:\Users\Administrator\Desktop\pol_DER.csv')