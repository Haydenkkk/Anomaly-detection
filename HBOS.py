import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 加载数据
df = pd.read_csv("Superstore.csv")

# 选择特征
X = df[['Sales', 'Profit']]

# 将特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义 HBOS 函数
def hbos(X, bins=10):
    # 将特征划分为 bins 个区间
    digitized = np.digitize(X, bins=np.linspace(np.min(X),np.max(X), bins - 1))
    # 计算每个区间的频率
    hist, _ = np.histogramdd(digitized, bins=(bins, bins))
    hist = hist / hist.sum()
    # print(hist[1][1])
    for i in range(bins):
        for j in range(bins):
            print(i,j,hist[i][j])
    # 计算每个实例的 HBOS 值
    # print(hist)
    scores = []
    for i in digitized:
        p1 = 0
        p2 = 0
        for j in range(bins):
            p1 += hist[i[0]][j]
            p2 += hist[j][i[1]]
        # p1 = np.sum(hist[i[0], :])
        # p2 = np.sum(hist[:, i[1]])
        p = p1 * p2
        # p = hist[i[0]][i[1]]
        if p == 0:
            p = 1e-10
        scores.append(np.log(1 / p))
    # print(scores)
    return scores

# 计算异常值分数
scores_pred = hbos(X_scaled)

# 设置异常值阈值
threshold = np.percentile(scores_pred, 99)
print(threshold)

# 标记异常值
y_pred = scores_pred > threshold
n_outliers = np.sum(y_pred)
outlier_indices = np.where(y_pred == 1)[0]


plt.figure(figsize=(10, 8))
# 正常值的散点图
b = sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], edgecolor='white')
# 异常值的散点图
c = sns.scatterplot(x=X_scaled[y_pred == 1, 0], y=X_scaled[y_pred == 1, 1])

plt.axis('tight')
plt.title('HBOS', fontsize=20)

print('异常值数量: ', n_outliers)
print('所有异常值的序号: \n', outlier_indices)
plt.savefig('img/HBOS.png', dpi=300)
