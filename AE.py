import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv("Superstore.csv")
X = df[['Sales', 'Profit']]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# 转换为 PyTorch Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 定义 Autoencoder 模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 初始化模型
input_dim = X_train.shape[1]
hidden_dim = 2  # 低维的隐变量维度
autoencoder = Autoencoder(input_dim, hidden_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 训练 Autoencoder
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = autoencoder(X_train_tensor)

    # 计算损失
    loss = criterion(outputs, X_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试集上的重构误差
with torch.no_grad():
    test_outputs = autoencoder(X_test_tensor)
    test_loss = criterion(test_outputs, X_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# 使用训练好的 Autoencoder 进行异常检测
reconstruction_errors = np.linalg.norm(X_test - test_outputs.numpy(), axis=1)
threshold = np.percentile(reconstruction_errors, 96)

# 标记异常点
anomalies = X_test[reconstruction_errors > threshold]

# 绘制图表
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], label='Normal')
sns.scatterplot(x=anomalies[:, 0], y=anomalies[:, 1], label='Anomaly')
plt.title('Autoencoder Anomaly Detection')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.legend()
plt.show()
