import torch
from torch.utils.data import Dataset,DataLoader, DataLoader
import numpy as np
import pandas as pd
import os
import tqdm
def process_srs_file(folder_path):
    # 定义文件夹路径
    # folder_path = r'20250530'  # 替换为你的文件夹路径

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
    return result_array

# position_to_coordinates = {
#     1: (0.0, 120.0),
#     2: (120.0, 120.0),
#     3: (240.0, 120.0),
#     4: (360.0, 120.0),
#     5: (360.0, 240.0),
#     6: (240.0, 240.0),
#     7: (120.0, 240.0),
#     8: (0.0, 240.0),
#     9: (-120.0, 240.0),
#     10: (-240.0, 240.0),
#     11: (-180.0, 360.0),
#     12: (-180.0, 480.0),
#     13: (-180.0, 600.0),
#     14: (-180.0, 720.0),
#     15: (-60.0, 360.0),
#     16: (60.0, 360.0),
#     17: (180.0, 360.0),
#     18: (300.0, 360.0),
#     19: (300.0, 480.0),
#     20: (420.0, 480.0),
#     21: (420.0, 600.0),
#     22: (300.0, 600.0),
#     23: (180.0, 600.0),
#     24: (60.0, 600.0),
#     25: (-60.0, 600.0),
#     26: (-60.0, 480.0),
#     27: (60.0, 480.0),
#     28: (-60.0, 720.0),
#     29: (60.0, 720.0),
#     30: (180.0, 720.0),
#     31: (180.0, 840.0),
#     32: (60.0, 840.0),
#     33: (-60.0, 840.0),
#     34: (-180.0, 900.0),
#     35: (-300.0, 1020.0),
#     36: (-180.0, 1020.0),
#     37: (-60.0, 960.0),
#     38: (60.0, 960.0)
# }

position_to_coordinates = {
    1: (360.0, 120.0),
    2: (180.0, 360.0),
    3: (0.0, 120.0),
    4: (180.0, 600.0),
    5: (60.0, 840.0)
}
for position in position_to_coordinates:
    x, y = position_to_coordinates[position]
    scale = 0.0001  # 修改坐标的放缩倍数
    position_to_coordinates[position] = (scale*x, scale*y)  
# 定义Dataset类
class MyDataSet(Dataset):
    def __init__(self, folder_path):
        print("fold path is ",folder_path)
        self.data_total = process_srs_file(folder_path)

        # valid_len = (self.data_total.shape[0] // 3) * 3
        # print("valid_len is", valid_len)
        print("origin_len is", self.data_total.shape[0])

        # self.data_total = self.data_total[:valid_len]
        
        self.data_total = self.data_total.reshape(-1, 1, 32,4)     
        
        # 对特征数据进行归一化
        feature_data = self.data_total[:,:,:,1:]
        self.feature_mean = np.mean(feature_data)
        self.feature_std = np.std(feature_data)
        self.data_total[:,:,:,1:] = (feature_data - self.feature_mean) / self.feature_std
        
        self.len = self.data_total.shape[0]
        
        print('='*40)
        print('输入数据总维度为 ', self.data_total[:,:,:,1:].shape)  # (N, 32, 256, 8)
        print('标签总维度为 ', self.data_total[:,:,:,0].shape)     # (N, 256)
        print('='*40)
 
        # 保存坐标映射
        self.position_to_coordinates = position_to_coordinates

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # 特征部分：天线数据
        features = torch.FloatTensor(self.data_total[index, :, :, 1:])  # (N, 32, 256, 8)
        
        # 获取物理坐标作为标签（通过第一列的序号映射到坐标）
        position_id = int(self.data_total[index, 0, 0, 0])  # 获取位置序号
        label = self.position_to_coordinates[position_id]  # 获取对应的坐标值
        
        # 将坐标值转为Tensor并返回
        label = torch.FloatTensor(label)  # 假设是二维坐标 [x, y]
        
        return features, label

if __name__ == "__main__":
    folder_path = r'20250530'

    dataset = MyDataSet(folder_path=folder_path)
    # 创建DataLoader对象
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=2,  #weight_decay=1e-3
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=8)
    # 读取一个batch的数据
    print("打印前几个数据样本：")
    for i, (features, labels) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print("Features shape:", features.shape)  # (batch_size, 32, 256, 8)
        print("Labels shape:", labels.shape)  # (batch_size, 2) 这里假设是二维坐标 (x, y)
        # print("Features (first sample):", features[0, :, :, :])  # 打印第一个样本的特征
        print("Label (first sample):", labels[0])  # 打印第一个样本的标签
        # print("Features (second sample):", features[1, :, :, :])  # 打印第二个样本的特征
        print("Label (second sample):", labels[1])  # 打印第二个样本的标签
        print("-" * 40)

        # 如果只想打印一次前几个样本，break退出循环
        if i==5:
            break

