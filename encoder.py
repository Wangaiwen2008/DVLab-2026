# 最简单的 OKTI 编码器：把一串 RGB 颜色变成 p 像素格式

def encode_okti(pixels, filename):
    # 假设我们固定写一个 2x2 的图片
    width = 2
    height = 2

    with open(filename, 'w') as f:
        # 1. 写入头部
        f.write("okti\n")
        f.write(f"{width} {height}\n")

        # 2. 遍历每个像素，转成十六进制
        for color in pixels:
            r, g, b = color
            # {:02x} 的意思是：转成 16 进制，如果只有 1 位就在前面补 0
            hex_str = f"p{r:02x}{g:02x}{b:02x}\n"
            f.write(hex_str)
            print(f"编码了一个像素: {hex_str.strip()}")


if __name__ == '__main__':
    # 这是我们要压缩的原始数据（对应 tiny.okti）
    my_pixels = [
        (255, 0, 0),  # 纯红
        (255, 255, 255),  # 纯白
        (0, 255, 0),  # 纯绿
        (0, 0, 255)  # 纯蓝
    ]

    # 运行编码器，输出到 output.okti
    encode_okti(my_pixels, 'output.okti')
    print("编码完成！请查看 output.okti 文件。")