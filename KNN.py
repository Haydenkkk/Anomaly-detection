import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def knn_outlier_detection(data, k, aggregation='mean'):
    num_samples = data.shape[0]
    distances = np.zeros((num_samples, num_samples))

    # 计算欧氏距离
    # for i in range(num_samples):
    #     for j in range(i + 1, num_samples):
    #             distances[i, j] = np.linalg.norm(data[i] - data[j])
    #             distances[j, i] = distances[i, j]
    
    # distances = cdist(data, data)

    neighbors_model = NearestNeighbors(n_neighbors=k)
    neighbors_model.fit(data)
    distances, _ = neighbors_model.kneighbors(data)

    # 排序选择最近的k个邻居
    sorted_distances = np.sort(distances, axis=1)
    k_nearest_neighbors = sorted_distances[:, 1:k+1]
    # 指定方法聚合
    if aggregation == 'mean':
        aggregated_distances = np.mean(k_nearest_neighbors, axis=1)
    elif aggregation == 'median':
        aggregated_distances = np.median(k_nearest_neighbors, axis=1)
    elif aggregation == 'max':
        aggregated_distances = np.max(k_nearest_neighbors, axis=1)
    else:
        raise ValueError("选择 'mean'、'median' 或 'max'")    
    return aggregated_distances

# 数据准备
df = pd.read_csv("Superstore.csv")
data_to_analyze = df[['Sales', 'Profit']].copy()

# 数据标准处理
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_to_analyze)

# 计算异常值
aggregated_distances = knn_outlier_detection(scaled_data, k=5, aggregation='mean')
data_to_analyze['AggregatedDistances'] = aggregated_distances

# 设定异常点的阈值
threshold = np.percentile(aggregated_distances, 99)

# 标记异常点
data_to_analyze['IsOutlier'] = data_to_analyze['AggregatedDistances'] > threshold

# 获取所有异常值的索引
outlier_indices = data_to_analyze.index[data_to_analyze['IsOutlier']].tolist()


print('异常值个数:', len(outlier_indices), '所有异常值的序号: \n', outlier_indices)
# plt.style.use('Solarize_Light2')
plt.figure(figsize=(10, 8))
sns.scatterplot(x="Sales", y="Profit", hue='IsOutlier', data=data_to_analyze)

plt.legend(loc='best')
plt.savefig('img/KNN.png', dpi=300)