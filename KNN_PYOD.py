import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from pyod.models.knn import KNN
from scipy import stats
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

# 数据准备
df = pd.read_csv("Superstore.csv")
X = df[['Sales', 'Profit']]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# print(X_scaled)
# 模型拟合
outliers_fraction = 0.01
clf = KNN(contamination=outliers_fraction)
clf.fit(X_scaled)

# 计算异常值
y_pred = clf.predict(X_scaled)
n_inliers = len(y_pred) - np.count_nonzero(y_pred)
n_outliers = np.count_nonzero(y_pred == 1)

# 获取所有异常值的索引
outlier_indices = np.where(y_pred == 1)[0]

# 输出所有异常值的序号
print('异常值数量: ', n_outliers)
print('所有异常值的序号: \n',outlier_indices)

# 绘制图表
plt.figure(figsize=(10, 8))
# 绘制异常值和正常值的散点图
b = sns.scatterplot(x=X_scaled[y_pred == 0, 0], y=X_scaled[y_pred == 0, 1], edgecolor='white')
c = sns.scatterplot(x=X_scaled[y_pred == 1, 0], y=X_scaled[y_pred == 1, 1], edgecolor='white')
plt.axis('tight')
plt.title('PYOD_KNN', fontsize=20)

plt.savefig('img/KNN_PYOD.png',dpi = 300)