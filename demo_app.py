import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn


# 1. 网络结构（必须和训练时一模一样）
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 加载模型
model = CNNNet()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()


# 2. 界面和画板逻辑
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别 Demo (CNN增强版)")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white', cursor='cross')
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.btn_predict = tk.Button(root, text="识别数字", command=self.predict, font=("Arial", 14))
        self.btn_predict.pack(side=tk.LEFT, padx=20, pady=10)

        self.btn_clear = tk.Button(root, text="清空画板", command=self.clear, font=("Arial", 14))
        self.btn_clear.pack(side=tk.RIGHT, padx=20, pady=10)

        self.label_result = tk.Label(root, text="请在画板上写数字", font=("Arial", 16), fg="blue")
        self.label_result.pack(pady=10)

        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        r = 6
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="请在画板上写数字", fg="blue")

    def predict(self):
        bbox = self.image.getbbox()
        if not bbox:
            return

        img = self.image.crop(bbox)

        w, h = img.size
        if w > h:
            new_w = 20
            new_h = int(h * (20 / w))
        else:
            new_h = 20
            new_w = int(w * (20 / h))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        img_new = Image.new("L", (28, 28), color=255)
        img_new.paste(img, ((28 - new_w) // 2, (28 - new_h) // 2))

        img_binary = img_new.point(lambda x: 0 if x < 128 else 255, '1')

        img_inverted = ImageOps.invert(img_binary.convert("L"))
        tensor = torch.tensor(list(img_inverted.getdata()), dtype=torch.float32).view(1, 1, 28, 28) / 255.0

        with torch.no_grad():
            output = model(tensor)
            _, predicted = torch.max(output.data, 1)
            prob = torch.softmax(output, dim=1)[0][predicted[0]].item()

        self.label_result.config(text=f"识别结果: {predicted.item()} (概率: {prob:.2f})", fg="green")


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("350x450")
    app = App(root)
    root.mainloop()