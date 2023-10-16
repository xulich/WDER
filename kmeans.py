import pandas as pd
import numpy as np

# 定义欧式距离
def euclidean_distance(x1, x2, w):
    distance = 0
    # 距离的平方项再开根号
    for i in range(len(x1)):
        distance += w[i]*pow((x1[i] - x2[i]), 2)
    return np.sqrt(distance)

def distance(a, b, w):
    dis =[]
    for i in range(len(a)):
        dis.append(euclidean_distance(a.iloc[i,].values,b, w))
    return dis

def centroids_init(k, X):
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    for i in range(k):
        # 每一次循环随机选择一个类别中心
        centroid = X[np.random.choice(range(n_samples))]
        centroids[i] = centroid
    return centroids

def closest_centroid(sample, centroids, w):
    closest_i = 0
    closest_dist = float('inf')
    for i, centroid in enumerate(centroids):
        # 根据欧式距离判断，选择最小距离的中心点所属类别
        distance = euclidean_distance(sample, centroid, w)
        if distance < closest_dist:
            closest_i = i
            closest_dist = distance
    return closest_i


def create_clusters(centroids, k, X, w):
    n_samples = np.shape(X)[0]
    clusters = [[] for _ in range(k)]
    for sample_i, sample in enumerate(X):
        # 将样本划分到最近的类别区域
        centroid_i = closest_centroid(sample, centroids, w)
        clusters[centroid_i].append(sample_i)
    return clusters

# 根据上一步聚类结果计算新的中心点
def calculate_centroids(clusters, k, X):
    n_features = np.shape(X)[1]
    centroids = np.zeros((k, n_features))
    # 以当前每个类样本的均值为新的中心点
    for i, cluster in enumerate(clusters):
        centroid = np.mean(X[cluster], axis=0)
        centroids[i] = centroid
    return centroids

# 获取每个样本所属的聚类类别
def get_cluster_labels(clusters, X):
    y_pred = np.zeros(np.shape(X)[0])
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred

# 根据上述各流程定义kmeans算法流程
def kmeans(X, k, max_iterations, w):
    # 1.初始化中心点
    centroids = centroids_init(k, X)
    # 遍历迭代求解
    for _ in range(max_iterations):
        # 2.根据当前中心点进行聚类
        clusters = create_clusters(centroids, k, X, w)
        # 保存当前中心点
        prev_centroids = centroids
        # 3.根据聚类结果计算新的中心点
        centroids = calculate_centroids(clusters, k, X)
        # 4.设定收敛条件为中心点是否发生变化
        diff = centroids - prev_centroids
        if not diff.any():
            break
    # 返回最终的聚类标签
    return get_cluster_labels(clusters, X), centroids