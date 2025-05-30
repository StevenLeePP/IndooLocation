import numpy as np
import os

# 定义文件夹路径
folder_path = r'20250530'  # 替换为你的文件夹路径

# 初始化一个空的 NumPy 数组，用于存储最终结果
result_array = np.empty((0, 4), dtype=np.int16)

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件扩展名是否为 .bin
    if file_name.endswith('.bin'):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, file_name)
        # 从文件名中提取数字作为标签
        label_str = file_name[8:10]  # 提取第 8 和第 9 个字符
        label = int(label_str)  # 将字符串转换为整数

        # 打开 .bin 文件并读取数据
        data = np.fromfile(file_path, dtype=np.int16)

        # 计算需要的行数
        num_rows = len(data) // 3

        # 如果数据不能整除3，截取前 num_rows * 3 个数据
        if len(data) % 3 != 0:
            data = data[:num_rows * 3]

        # 将数据重新排列成二维数组，每行3个 int16
        two_dim_array = data.reshape(-1, 3)

        # 创建标签列，标签值为提取的数字
        label_column = np.full((num_rows, 1), label, dtype=np.int16)

        # 将标签列添加到二维数组的最左侧
        temp_array = np.hstack((label_column, two_dim_array))

        # 将当前文件的结果纵向拼接到最终结果数组中
        result_array = np.vstack((result_array, temp_array))

# 打印最终结果数组的形状
print("最终结果数组的形状为:", result_array.shape)

# 打印最终结果数组的前几行
print("最终结果数组的前几行:")
print(result_array[:200])
print("最终结果数组的后几行:")
print(result_array[-10:-1])
