import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 数据准备：把图片转成张量，并下载到本地 data 文件夹
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 2. 定义神经网络：一个最简单的全连接网络（输入784个像素，输出10个类别）
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 28x28的图片展平成784
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # 输出0-9这10个数字的概率

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平图片
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 3. 开始训练
model = SimpleNet()
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

print("开始训练模型，请稍候...")
for epoch in range(10):  # 只训练2轮，又快又好
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # 每处理300个批次，打印一下进度
        if batch_idx % 300 == 0:
            print(f"训练轮次: {epoch + 1}, 当前批次: {batch_idx}, 当前Loss: {loss.item():.4f}")

# 4. 测试准确率
print("开始测试...")
model.eval()  # 切换到测试模式
correct = 0
total = 0
with torch.no_grad():  # 测试不需要计算梯度
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"最终测试集准确率: {100 * correct / total:.2f}%")
torch.save(model.state_dict(), 'mnist_model.pth')
