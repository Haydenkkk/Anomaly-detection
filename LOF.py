import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv("Superstore.csv")

# 选择特征
X = df[['Sales', 'Profit']]

# 将特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def calculate_reachability_density(X, k):
    n = X.shape[0]
    lrd = np.zeros(n)

    for i in range(n):
        distances = pairwise_distances(X[i].reshape(1, -1), X).flatten()
        neighbors_indices = np.argsort(distances)[1:k+1]  # 排除自身，选择最近的 k 个邻居

        # 检查分母是否为零
        denominator = np.sum(distances[neighbors_indices])
        if denominator == 0:
            lrd[i] = np.nan
        else:
            lrd[i] = 1 / (denominator / k)

    return lrd

def calculate_local_outlier_factor(X, k, lrd):
    n = X.shape[0]
    lof = np.zeros(n)

    for i in range(n):
        distances = pairwise_distances(X[i].reshape(1, -1), X).flatten()
        neighbors_indices = np.argsort(distances)[1:k+1]  # 排除自身，选择最近的 k 个邻居

        # 检查局部可达密度是否为零
        if lrd[i] == 0 or np.any(lrd[neighbors_indices] == 0):
            lof[i] = np.nan
        else:
            lof[i] = np.sum(lrd[neighbors_indices]) / (lrd[i] * k)

    return lof

def identify_outliers(lof, threshold):
    outliers_indices = np.where(lof > threshold)[0]
    return outliers_indices

# LOF 参数
k = 100  # 邻近点的数量
threshold = 2
# 计算 LOF
lrd = calculate_reachability_density(X_scaled, k)
lof = calculate_local_outlier_factor(X_scaled, k, lrd)

# 设置异常点的比例为 0.01
threshold = np.percentile(lof, 99)

# 标识异常点
outliers_indices = identify_outliers(lof, threshold)

# 绘制图表
plt.figure(figsize=(10, 8))
b = sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], edgecolor='white')
c = sns.scatterplot(x=X_scaled[outliers_indices, 0], y=X_scaled[outliers_indices, 1])
plt.axis('tight')
plt.title('Local Outlier Factor (LOF)', fontsize=20)


# 输出结果
# print("局部可达密度 (LRD):", lrd)
# print("局部异常因子 (LOF):", lof)
print("异常点的数量:", len(outliers_indices))
print("异常点的索引:\n", outliers_indices)
# print("LOF 分位数阈值:", threshold)
plt.savefig('img/LOF.png',dpi = 300)