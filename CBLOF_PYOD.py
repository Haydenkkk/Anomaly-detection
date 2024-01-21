import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pyod.models.cblof import CBLOF

# 加载数据
df = pd.read_csv("Superstore.csv")

# 选择特征
X = df[['Sales', 'Profit']]

# 将特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设定异常值比例
outliers_fraction = 0.01

# 定义CBLOF模型
clf = CBLOF(contamination=outliers_fraction, check_estimator=False, random_state=0)

# 训练模型
clf.fit(X_scaled)

# 预测异常值
y_pred = clf.predict(X_scaled)

# 获取所有异常值的索引
outlier_indices = np.where(y_pred == 1)[0]

# 输出所有异常值的序号
print('所有异常值的序号: \n', outlier_indices)

# 绘制图表
plt.figure(figsize=(10, 8))
# 绘制正常值的散点图
b = sns.scatterplot(x=X_scaled[y_pred == 0, 0], y=X_scaled[y_pred == 0, 1], edgecolor='white')
# 绘制异常值的散点图
c = sns.scatterplot(x=X_scaled[y_pred == 1, 0], y=X_scaled[y_pred == 1, 1], edgecolor='white')
plt.axis('tight')
plt.title('CBLOF_PYOD', fontsize=20)

print('异常值数量: ', len(outlier_indices))
plt.savefig('img/CBLOF_PYOD.png',dpi = 300)
