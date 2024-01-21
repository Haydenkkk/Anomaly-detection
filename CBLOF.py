import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv("Superstore.csv")

# 选择特征
X = df[['Sales', 'Profit']]

# 将特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设定异常值比例
outliers_fraction = 0.01

# 定义KMeans模型
kmeans = KMeans(n_clusters=8)
kmeans.fit(X_scaled)

# 预测每个数据点的簇标签
labels = kmeans.labels_

# 计算每个簇的中心点
centers = kmeans.cluster_centers_

# 计算每个点到其簇中心的距离
distances = np.linalg.norm(X_scaled - centers[labels], axis=1)

# 根据异常值比例找到阈值
threshold = np.percentile(distances, 100 * (1 - outliers_fraction))

# 找到异常值的索引
outlier_indices = np.where(distances > threshold)[0]

# 输出所有异常值的序号
print('所有异常值的序号: \n', outlier_indices)

# 绘制图表
plt.figure(figsize=(10, 8))

# 绘制正常值的散点图
b = sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], edgecolor='white')

# 绘制异常值的散点图
c = sns.scatterplot(x=X_scaled[outlier_indices, 0], y=X_scaled[outlier_indices, 1])

plt.axis('tight')
plt.title('CBLOF', fontsize=20)

print('异常值数量: ', len(outlier_indices))
plt.savefig('img/CBLOF.png',dpi = 300)