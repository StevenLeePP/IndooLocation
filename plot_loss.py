import matplotlib.pyplot as plt

# 定义一个函数来读取文件中的数值
def read_values_from_file(file_path):
    values = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()  # 去除多余的空格和换行符
            # 提取每一行的数字值
            value = float(line.split(":")[-1].strip())  # 提取冒号后的部分并转换为浮点数
            values.append(value)
    return values

# 读取 loss 和 accuracy 的数据
loss_values = read_values_from_file("loss_test.txt")
accuracy_values = read_values_from_file("accuracy_test.txt")

# 创建一个图形窗口
plt.figure(figsize=(10, 12))

# 绘制 Loss 图
plt.subplot(2, 1, 1)  # 第一个子图
iterations = range(1, len(loss_values) + 1)  # 从1开始的迭代次数
plt.plot(iterations, loss_values, marker='o', linestyle='-', color='r', label='Loss')
plt.title("Loss Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.xticks(iterations)  # 设置横坐标为正整数
plt.grid(True)
plt.legend()

# 绘制 Accuracy 图
plt.subplot(2, 1, 2)  # 第二个子图
iterations = range(1, len(accuracy_values) + 1)  # 从1开始的迭代次数
plt.plot(iterations, accuracy_values, marker='o', linestyle='-', color='b', label='Accuracy')
plt.title("Accuracy Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.xticks(iterations)  # 设置横坐标为正整数
plt.grid(True)
plt.legend()

# 显示图形
plt.tight_layout()  # 自动调整子图间距
# plt.show()

plt.savefig("./accuracy_save/2.jpg")
print("绘图完毕")



# import matplotlib.pyplot as plt

# # 定义一个函数来读取文件中的数值
# def read_values_from_file(file_path):
#     values = []
#     with open(file_path, "r") as file:
#         for line in file:
#             line = line.strip()  # 去除多余的空格和换行符
#             # 提取每一行的数字值
#             value = float(line.split(":")[-1].strip())  # 提取冒号后的部分并转换为浮点数
#             values.append(value)
#     return values

# # 读取 loss 和 accuracy 的数据
# loss_values = read_values_from_file("E:\北邮2024\本科毕设\Model\lap\loss_test.txt")
# accuracy_values = read_values_from_file("E:\北邮2024\本科毕设\Model\ResNet\loss_test.txt")


# # 确保两个数据集的长度一致（如果需要）
# min_length = min(len(loss_values), len(accuracy_values))
# loss_values = loss_values[:min_length]
# accuracy_values = accuracy_values[:min_length]

# # 创建一个图形窗口
# plt.figure(figsize=(10, 6))

# # 从1开始的迭代次数
# iterations = range(1, min_length + 1)

# # 绘制 Loss 图
# plt.plot(iterations, loss_values, marker='o', linestyle='-', color='r', label='Normal')

# # 绘制 Accuracy 图
# plt.plot(iterations, accuracy_values, marker='o', linestyle='-', color='b', label='ResNet')

# # 设置标题和标签
# plt.title("Loss Over Epochs")
# plt.xlabel("Iteration")
# plt.ylabel("Value")
# plt.xticks(iterations)  # 设置横坐标为正整数
# plt.grid(True)
# plt.legend()

# # 显示图形
# plt.tight_layout()  # 自动调整子图间距
# plt.show()



# loss_values = read_values_from_file(r"E:\北邮2024\本科毕设\Model\lap\accuracy_test.txt")
# accuracy_values = read_values_from_file(r"E:\北邮2024\本科毕设\Model\ResNet\accuracy_test.txt")


# # 确保两个数据集的长度一致（如果需要）
# min_length = min(len(loss_values), len(accuracy_values))
# loss_values = loss_values[:min_length]
# accuracy_values = accuracy_values[:min_length]

# # 创建一个图形窗口
# plt.figure(figsize=(10, 6))

# # 从1开始的迭代次数
# iterations = range(1, min_length + 1)

# # 绘制 Loss 图
# #plt.plot(iterations, loss_values, marker='o', linestyle='-', color='r', label='Normal')

# # 绘制 Accuracy 图
# plt.plot(iterations, accuracy_values, marker='o', linestyle='-', color='b', label='ResNet')

# # 设置标题和标签
# plt.title("Accuracy Over Epochs")
# plt.xlabel("Iteration")
# plt.ylabel("Value")
# plt.ylim((0,100))
# plt.xticks(iterations)  # 设置横坐标为正整数
# plt.grid(True)
# plt.legend()

# # 显示图形
# plt.tight_layout()  # 自动调整子图间距
# plt.show()