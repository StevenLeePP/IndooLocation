import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 假设你的CSV文件名为 'data.csv'
file_path = 'train_predictions.csv'

# 读取CSV文件，不将第一行作为表头处理
df = pd.read_csv(file_path, header=None)

# 转换为NumPy数组
numpy_array = df.to_numpy()
numpy_array = numpy_array[:, 1:]

# 强制转换为float类型
try:
    numpy_array = numpy_array.astype(float)
except ValueError as e:
    print("数据类型转换失败，请检查CSV文件中的数据是否为数值类型。错误信息：", e)
    exit()

# 将所有原始数据乘以10000
numpy_array *= 10000

# 计算第一列数据减去第三列数据的平方，加上第二列数据减去第四列数据的平方
result = (numpy_array[:, 0] - numpy_array[:, 2]) ** 2 + (numpy_array[:, 1] - numpy_array[:, 3]) ** 2
result_sqrt = np.sqrt(result)
# # 将计算结果作为第五列添加到 NumPy 数组中
# numpy_array = np.hstack((numpy_array, result[:, np.newaxis]))

# # 打印完整的NumPy数组
# print("完整的NumPy数组（包含计算结果作为第五列）：")
# print(numpy_array)

# 绘制 result 的值
plt.figure(figsize=(10, 6))
plt.plot(result_sqrt,  linestyle='-', color='b')
plt.title('Result Values')
plt.xlabel('Index')
plt.ylabel('Result')
plt.grid(True)

# 保存图像
plt.savefig('result_plot.png')
plt.show()