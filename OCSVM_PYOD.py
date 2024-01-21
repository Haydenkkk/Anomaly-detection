import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyod.models.ocsvm import OCSVM
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv("Superstore.csv")

# 选择特征
X = df[['Sales', 'Profit']]

# 将特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建并训练 One-Class SVM 模型
ocsvm_model = OCSVM(contamination=0.01)  # contamination 表示异常值的比例
ocsvm_model.fit(X_scaled)

# 获取每个数据点的异常值分数
scores_pred = ocsvm_model.decision_function(X_scaled)

# 预测异常值
y_pred = ocsvm_model.predict(X_scaled)
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
plt.title('One-Class SVM', fontsize=20)

print('异常值数量: ', n_outliers)
print('所有异常值的序号: \n', outlier_indices)
plt.savefig('img/OCSVM_PYOD.png',dpi = 300)
