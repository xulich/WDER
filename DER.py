import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
import sklearn.tree as st
from kmeans import  kmeans
from kmeans import distance

def DER(X_train, y_train, X_test,  n_clusters, beta, max_iterations, eta, alpha, model_name):
    dtr = st.DecisionTreeRegressor()
    dtr.fit(X_train, y_train)
    importance = dtr.feature_importances_
    w = np.exp(importance * beta)

    labels, centroids = kmeans(X_train.values, n_clusters, max_iterations, w)

    #采样
    bootstrap_num = int(X_train.shape[0] * eta)
    clustered_indices = []
    for i in range(n_clusters):
        sample_centroid_distances = distance(X_train, centroids[i], w)
        index_i = np.argsort(sample_centroid_distances)[:bootstrap_num]
        clustered_indices.append(index_i)

    # 模型构建
    classifier_list = []
    for clustered_index in clustered_indices:
        X_train_sub = X_train.values[clustered_index, :]
        y_train_sub = y_train.values[clustered_index]

        models_list = {
            'SVR': SVR(kernel='rbf', C=0.8, gamma=0.2),
            'XGB': xgb.XGBRFRegressor(),
            'GBRT': GradientBoostingRegressor(),
            'RFR': RandomForestRegressor(),
            'Enet': ElasticNetCV(),
            'Lasso': Lasso()
        }

        model = models_list[model_name]
        model.fit(X_train_sub, y_train_sub)
        classifier_list.append(model)

    # 结果集成
    result_matrix = pd.DataFrame()
    for i, clustered_index in enumerate(clustered_indices):
        result_matrix["h" + str(i)] = classifier_list[i].predict(X_test)

    t = []
    for i in range(len(X_test)):
        t.append(distance(pd.DataFrame(centroids), X_test.iloc[i,].values, w))

    weight_matrix = 1 / np.exp(np.array(t) * alpha)
    norm_weight_matrix = weight_matrix / weight_matrix.sum(1).reshape(-1, 1)
    final_result_matraix = (norm_weight_matrix * result_matrix).sum(1)

    return(final_result_matraix)
