import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 假设你的CSV文件名为 'train_predictions.csv'
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
print("origin is ", numpy_array[:5])

# 计算第一列数据减去第三列数据的平方，加上第二列数据减去第四列数据的平方
result = (numpy_array[:, 0] - numpy_array[:, 2]) ** 2 + (numpy_array[:, 1] - numpy_array[:, 3]) ** 2
result_sqrt = np.sqrt(result)

# 计算平均值、中位数和90%分位数
mean_value = np.mean(result_sqrt)
median_value = np.median(result_sqrt)
cep_90_value = np.percentile(result_sqrt, 90)  # 计算90%分位数

# 绘制 result 的值
plt.figure(figsize=(10, 6))
plt.plot(result_sqrt, color='b', label='Euclidean Distance')
# plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean Value: {mean_value:.2f}')
plt.axhline(y=median_value, color='g', linestyle='-.', label=f'CEP50%: {median_value:.2f}')
plt.axhline(y=cep_90_value, color='purple', linestyle=':', label=f'CEP90%: {cep_90_value:.2f}')
plt.title('Euclidean Distance Over Index')
plt.xlabel('Index')
plt.ylabel('Euclidean Distance')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig('result_plot_with_lines.png')
plt.show()

print("图片保存完毕")