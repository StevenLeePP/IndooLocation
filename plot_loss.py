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
loss_test = read_values_from_file("loss_test.txt")
accuracy_test = read_values_from_file("accuracy_test.txt")
loss_train = read_values_from_file("loss.txt")
accuracy_train = read_values_from_file("accuracy.txt")

# 创建一个图形窗口
plt.figure(figsize=(10, 12))

# 绘制 Train-Loss 图
plt.subplot(2, 1, 1)  # 第一个子图
iterations = range(1, len(loss_train) + 1)  # 从1开始的迭代次数
plt.plot(iterations, loss_train, marker='o', linestyle='-', color='r', label='Loss')
plt.title("Train-Loss")
plt.xlabel("traintLoss")
plt.ylabel("Loss")
plt.xticks(iterations)  # 设置横坐标为正整数
plt.grid(True)
plt.legend()

# 绘制 Test-Loss 图
plt.subplot(2, 1, 2)  # 第二个子图
iterations = range(1, len(loss_test) + 1)  # 从1开始的迭代次数
plt.plot(iterations, loss_test, marker='o', linestyle='-', color='b', label='Accuracy')
plt.title("Test-Loss")
plt.xlabel("Iteration")
plt.ylabel("testLoss")
plt.xticks(iterations)  # 设置横坐标为正整数
plt.grid(True)
plt.legend()

# 显示图形
plt.tight_layout()  # 自动调整子图间距
# plt.show()
output="6.jpg"
plt.savefig("./accuracy_save/"+output)
print(output+"绘图完毕")



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
# loss_test = read_values_from_file("E:\北邮2024\本科毕设\Model\lap\loss_test.txt")
# accuracy_test = read_values_from_file("E:\北邮2024\本科毕设\Model\ResNet\loss_test.txt")


# # 确保两个数据集的长度一致（如果需要）
# min_length = min(len(loss_test), len(accuracy_test))
# loss_test = loss_test[:min_length]
# accuracy_test = accuracy_test[:min_length]

# # 创建一个图形窗口
# plt.figure(figsize=(10, 6))

# # 从1开始的迭代次数
# iterations = range(1, min_length + 1)

# # 绘制 Loss 图
# plt.plot(iterations, loss_test, marker='o', linestyle='-', color='r', label='Normal')

# # 绘制 Accuracy 图
# plt.plot(iterations, accuracy_test, marker='o', linestyle='-', color='b', label='ResNet')

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



# loss_test = read_values_from_file(r"E:\北邮2024\本科毕设\Model\lap\accuracy_test.txt")
# accuracy_test = read_values_from_file(r"E:\北邮2024\本科毕设\Model\ResNet\accuracy_test.txt")


# # 确保两个数据集的长度一致（如果需要）
# min_length = min(len(loss_test), len(accuracy_test))
# loss_test = loss_test[:min_length]
# accuracy_test = accuracy_test[:min_length]

# # 创建一个图形窗口
# plt.figure(figsize=(10, 6))

# # 从1开始的迭代次数
# iterations = range(1, min_length + 1)

# # 绘制 Loss 图
# #plt.plot(iterations, loss_test, marker='o', linestyle='-', color='r', label='Normal')

# # 绘制 Accuracy 图
# plt.plot(iterations, accuracy_test, marker='o', linestyle='-', color='b', label='ResNet')

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