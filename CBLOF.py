import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# 23
def perform_cblof(X, alpha=0.9, beta=5, n_clusters=10):
    # 进行标准化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # 聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_scaled)

    # 预测每个数据点的簇标签
    labels = kmeans.labels_
    # 计算每个簇的中心点
    centers = kmeans.cluster_centers_
    print("Cluster centers:", centers)
    # 将簇按大小排序
    cluster_sizes = np.bincount(labels)
    sorted_clusters = np.argsort(cluster_sizes)[::-1]
    print("Cluster sizes:", cluster_sizes[sorted_clusters])
    # 找到边界Cb
    cumulative_size = 0
    for b in range(len(sorted_clusters) - 1):
        cumulative_size += cluster_sizes[sorted_clusters[b]]
        if (cumulative_size / len(X)) >= alpha or (cluster_sizes[sorted_clusters[b]] / cluster_sizes[sorted_clusters[b + 1]]) >= beta:
            break

    # 划分大簇和小簇
    large_cluster_indices = sorted_clusters[:b+1]
    small_cluster_indices = sorted_clusters[b+1:]
    for i in range(len(large_cluster_indices)):
        print("Large cluster", i, "size:", cluster_sizes[large_cluster_indices[i]])
    # 计算CBLOF
    distances = np.linalg.norm(X_scaled - centers[labels], axis=1)
    cblof_scores = np.zeros(len(X))
    
    for i in large_cluster_indices:
        cblof_scores[labels == i] = cluster_sizes[i] * distances[labels == i]

    for i in small_cluster_indices:
        min_distances_to_large_clusters = np.min(np.linalg.norm(centers[large_cluster_indices] - X_scaled[i], axis=1))
        cblof_scores[labels == i] = cluster_sizes[i] * min_distances_to_large_clusters


    # 根据异常值比例找到阈值
    threshold = np.percentile(cblof_scores, 1)

    # 找到异常值的索引
    outlier_indices = np.where(cblof_scores < threshold)[0]

    return outlier_indices

def plot_cblof_results(X, outlier_indices):
    # 绘制图表
    plt.figure(figsize=(10, 8))

    # 绘制正常值的散点图
    sns.scatterplot(x=X[:, 0], y=X[:, 1], edgecolor='white', label='Normal')

    # 绘制异常值的散点图
    sns.scatterplot(x=X[outlier_indices, 0], y=X[outlier_indices, 1], label='Outlier')

    plt.axis('tight')
    plt.title('CBLOF', fontsize=20)
    plt.legend()

    # 保存图表
    plt.savefig('img/CBLOF.png', dpi=300)
    # plt.show()



df = pd.read_csv("Superstore.csv")
X = df[['Sales', 'Profit']]

outlier_indices = perform_cblof(X)

print('异常值数量: ', len(outlier_indices))
print('所有异常值的序号: \n', outlier_indices)

plot_cblof_results(X.values, outlier_indices)


