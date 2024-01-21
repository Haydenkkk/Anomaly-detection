import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pyod.models.hbos import HBOS

# 加载数据
df = pd.read_csv("Superstore.csv")

# 选择特征
X = df[['Sales', 'Profit']]

# 将特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

outliers_fraction = 0.01
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
clf = HBOS(contamination=outliers_fraction)
clf.fit(X)

# 获取每个数据点的异常值分数
scores_pred = clf.decision_function(X)

# 预测异常值
y_pred = clf.predict(X)
n_inliers = len(y_pred) - np.count_nonzero(y_pred)
n_outliers = np.count_nonzero(y_pred == 1)
outlier_indices = np.where(y_pred == 1)[0]
# 绘制图表
plt.figure(figsize=(10, 8))

# 绘制正常值的散点图
b = sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], edgecolor='white')

# 绘制异常值的散点图
c = sns.scatterplot(x=X_scaled[y_pred == 1, 0], y=X_scaled[y_pred == 1, 1])

plt.axis('tight')
plt.title('HBOS', fontsize=20)

print('异常值数量: ', n_outliers)
print('所有异常值的序号: \n', outlier_indices)
plt.savefig('img/HBOS_PYOD.png',dpi = 300)