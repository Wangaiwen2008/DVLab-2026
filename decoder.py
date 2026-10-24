def decode_okti(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].strip()
    if first_line != 'okti':
        print("错误：不是 OKTI 文件！")
        return

    width, height = map(int, lines[1].strip().split())
    print(f"图片尺寸: 宽 {width}, 高 {height}")

    pixels = []
    previous_pixel = (0, 0, 0)
    colors_list = [(0, 0, 0)]

    def add_color_to_list(color):
        if color not in colors_list:
            colors_list.insert(0, color)
            if len(colors_list) > 256:
                colors_list.pop()

    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue

        if line.startswith('p'):
            r = int(line[1:3], 16)
            g = int(line[3:5], 16)
            b = int(line[5:7], 16)
            previous_pixel = (r, g, b)
            add_color_to_list(previous_pixel)
            pixels.append(previous_pixel)

        elif line.startswith('d'):
            r_diff = int(line[1], 16) - 8
            g_diff = int(line[2], 16) - 8
            b_diff = int(line[3], 16) - 8

            r = max(0, min(255, previous_pixel[0] + r_diff))
            g = max(0, min(255, previous_pixel[1] + g_diff))
            b = max(0, min(255, previous_pixel[2] + b_diff))

            previous_pixel = (r, g, b)
            add_color_to_list(previous_pixel)
            pixels.append(previous_pixel)

        elif line.startswith('r') or line.startswith('R'):
            if line.startswith('r'):
                count = int(line[1], 16)
            else:
                count = int(line[1:3], 16)

            for _ in range(count):
                pixels.append(previous_pixel)

        elif line.startswith('i') or line.startswith('I'):
            if line.startswith('i'):
                index = int(line[1], 16)
            else:
                index = int(line[1:3], 16)

            # 【新增：异常捕获】如果索引越界了，就报错退出
            try:
                previous_pixel = colors_list[index]
                pixels.append(previous_pixel)
                print(f"解析到一个索引像素: 使用了表里第 {index} 个颜色 RGB{previous_pixel}")
            except IndexError:
                print(f"错误：在读取 '{line}' 时，颜色索引 {index} 越界！当前历史记录表只有 {len(colors_list)} 个颜色。")
                return  # 直接结束函数

    print(f"一共解析了 {len(pixels)} 个像素！")


if __name__ == '__main__':
    decode_okti('test.okti')