import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn


# 1. 把咱们昨天的网络结构搬过来
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 加载你刚刚保存的模型
model = SimpleNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # 切换到测试模式


# 2. 界面和画板逻辑
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别 Demo")

        # 创建画板
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white', cursor='cross')
        self.canvas.pack(pady=10)

        # 绑定鼠标事件（按下开始画，拖动也画）
        self.canvas.bind("<B1-Motion>", self.paint)

        # 创建按钮
        self.btn_predict = tk.Button(root, text="识别数字", command=self.predict, font=("Arial", 14))
        self.btn_predict.pack(side=tk.LEFT, padx=20, pady=10)

        self.btn_clear = tk.Button(root, text="清空画板", command=self.clear, font=("Arial", 14))
        self.btn_clear.pack(side=tk.RIGHT, padx=20, pady=10)

        # 显示结果的标签
        self.label_result = tk.Label(root, text="请在画板上写数字", font=("Arial", 16), fg="blue")
        self.label_result.pack(pady=10)

        # 创建一个隐藏的 PIL 图像，用来保存画板上的内容
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        # 画笔大小
        r = 10
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        # 在界面上画黑线
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        # 在 PIL 图像上也画黑线（为了传给模型）
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="请在画板上写数字", fg="blue")

    def predict(self):
        # 1. 裁剪出数字部分
        bbox = self.image.getbbox()
        if not bbox:
            return
        img = self.image.crop(bbox)

        # 2. 调整大小并居中 (MNIST 要求 28x28)
        img = img.resize((20, 20), Image.Resampling.LANCZOS)
        img_new = Image.new("L", (28, 28), color=255)
        img_new.paste(img, ((28 - 20) // 2, (28 - 20) // 2))

        # 3. 反转颜色并归一化 (变成张量)
        img_inverted = ImageOps.invert(img_new)
        tensor = torch.tensor(list(img_inverted.getdata()), dtype=torch.float32).view(1, 1, 28, 28) / 255.0

        # 4. 预测
        with torch.no_grad():
            output = model(tensor)
            _, predicted = torch.max(output.data, 1)
            prob = torch.softmax(output, dim=1)[0][predicted[0]].item()

        self.label_result.config(text=f"识别结果: {predicted.item()} (概率: {prob:.2f})", fg="green")


# 运行程序
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("350x450")
    app = App(root)
    root.mainloop()