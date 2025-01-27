import os
import re

from PIL import Image

# 指定文件夹路径和输出GIF的路径
folder_path = 'inference/trial/84_model_train' # 指定你的文件夹路径
output_gif_path = 'inference/trial/84_model_train/animated.gif'  # 输出GIF文件的路径

target_size = (1000, 500)  # 设置目标图像尺寸


def numerical_sort(value):
    """ Extracts numbers from filenames for proper sorting. """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])  # Converts strings to integers
    return parts

# 初始化一个列表来存储帧
frames = []

# 遍历文件夹中的所有PNG文件，并按数字顺序排序
for filename in sorted(os.listdir(folder_path), key=numerical_sort):
    if filename.endswith(".png"):
        # 组合文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 打开图像
        image = Image.open(file_path)
        # 调整图像大小
        image = image.resize(target_size, Image.ANTIALIAS)
        # 添加到帧列表中
        frames.append(image)

# 保存帧为一个GIF文件
if frames:
    frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=1000, loop=0)

print("GIF created successfully. Saved at:", output_gif_path)

