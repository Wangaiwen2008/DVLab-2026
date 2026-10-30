import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 伪造一条“看不见的曲线”数据（假设这是一条正弦波浪线）
x = np.linspace(-3, 3, 100)  # 在 -3 到 3 之间生成 100 个点
y = np.sin(x) + 0.5 * np.cos(2 * x)  # 真实的隐藏曲线

# 转换成 PyTorch 的张量，并调整形状为 (100, 1)
X_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
Y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)


# 2. 搭一个简单的全连接网络（输入1维x，输出1维y）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 3. 训练函数，方便我们测试不同次数
def train_model(epochs, lr=0.01):
    model = Net()
    criterion = nn.MSELoss()  # 均方误差
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()

    return model, loss.item()


# 4. 正式开始测试并画图！
plt.figure(figsize=(12, 4))

# 测试 10 次、100 次、1000 次的训练效果
epochs_list = [10, 100, 1000]
for i, epochs in enumerate(epochs_list):
    model, final_loss = train_model(epochs=epochs, lr=0.01)

    # 用训练好的模型去预测
    with torch.no_grad():
        predicted = model(X_tensor).numpy()

    # 画图
    plt.subplot(1, 3, i + 1)
    plt.plot(x, y, label="真实数据 (True)", color="blue", linewidth=2)
    plt.plot(x, predicted, label=f"预测曲线 (训练{epochs}次)", color="red", linestyle="--")
    plt.title(f"训练 {epochs} 次 (Loss: {final_loss:.4f})")
    plt.legend()
    plt.ylim(-2, 2)

plt.tight_layout()
plt.show()