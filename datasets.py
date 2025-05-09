import torch
from torch.utils.data import Dataset,DataLoader, DataLoader
import numpy as np
import pandas as pd
import os
import tqdm
def process_srs_file(file_path):
    """
    处理二进制文件，提取数据并转换为复数形式的数组。

    参数:
        file_path (str): 二进制文件的路径。

    返回:
        processed_data (numpy.ndarray): 处理后的数组。
    """
# 提取文件名的最后两位数字作为标签
    file_name = os.path.basename(file_path)
    # 假设文件名格式为 "srs_get_XX.bin"，提取 "XX" 部分
    label = int(file_name.split('_')[-1].split('.')[0])  # 提取 "01" 并转换为整数
    # 打开二进制文件
    with open(file_path, "rb") as fid1:
        # 读取所有数据，假设数据类型为 uint16
        fdata = np.fromfile(fid1, dtype=np.uint16)

    # 计算周期长度
    cycle_length = 2048 * 2 * 4 + 1

    # 计算周期数量
    num_cycles = len(fdata) // cycle_length
    # 初始化一个空列表来存储处理后的数据
    processed_data = []

    # 对每个周期进行处理
    for i in range(num_cycles):
        # 提取一个周期的数据
        cycle_data = fdata[i * cycle_length : (i + 1) * cycle_length]
        cycle_data = cycle_data[:-1]  # 去掉最后一个元素

        # 初始化复数形式的数组
        antenna_data_complex = np.empty((2048, 8), dtype=np.uint16)

        # 填充复数形式的数组
        for j in range(4):
            antenna_data_complex[:, j * 2] = cycle_data[j * 2048 * 2 : (j + 1) * 2048 * 2 : 2]  # 实部
            antenna_data_complex[:, j * 2 + 1] = cycle_data[j * 2048 * 2 + 1 : (j + 1) * 2048 * 2 : 2]  # 虚部

        # 保留奇数行
        antenna_data_complex = antenna_data_complex[::2, :]
        

        # 将处理后的数据添加到列表中
        processed_data.append(antenna_data_complex)

    # 将列表中的所有数据合并成一个大数组
    processed_data = np.concatenate(processed_data, axis=0)
    label_column = np.full((processed_data.shape[0], 1), label, dtype=np.uint16)
    processed_data = np.hstack((label_column, processed_data))

    return processed_data

position_to_coordinates = {
    1: (0.0, 120.0),
    2: (120.0, 120.0),
    3: (240.0, 120.0),
    4: (360.0, 120.0),
    5: (360.0, 240.0),
    6: (240.0, 240.0),
    7: (120.0, 240.0),
    8: (0.0, 240.0),
    9: (-120.0, 240.0),
    10: (-240.0, 240.0),
    11: (-180.0, 360.0),
    12: (-180.0, 480.0),
    13: (-180.0, 600.0),
    14: (-180.0, 720.0),
    15: (-60.0, 360.0),
    16: (60.0, 360.0),
    17: (180.0, 360.0),
    18: (300.0, 360.0),
    19: (300.0, 480.0),
    20: (420.0, 480.0),
    21: (420.0, 600.0),
    22: (300.0, 600.0),
    23: (180.0, 600.0),
    24: (60.0, 600.0),
    25: (-60.0, 600.0),
    26: (-60.0, 480.0),
    27: (60.0, 480.0),
    28: (-60.0, 720.0),
    29: (60.0, 720.0),
    30: (180.0, 720.0),
    31: (180.0, 840.0),
    32: (60.0, 840.0),
    33: (-60.0, 840.0),
    34: (-180.0, 900.0),
    35: (-300.0, 1020.0),
    36: (-180.0, 1020.0),
    37: (-60.0, 960.0),
    38: (60.0, 960.0)
}
for position in position_to_coordinates:
    x, y = position_to_coordinates[position]
    scale = 0.0001  # 修改坐标的放缩倍数
    position_to_coordinates[position] = (scale*x, scale*y)  
# 定义Dataset类
class MyDataSet(Dataset):
    def __init__(self, folder_path):
        """
        遍历文件夹中的所有 .bin 文件，运行 process_binary_file 函数，并将结果数组纵向拼接。
        参数:
            folder_path (str): 文件夹路径。
        返回:
            combined_data (numpy.ndarray): 所有文件处理后的纵向拼接数组。
        """
        # 初始化一个空列表来存储每个文件的处理结果
        all_processed_data = []

        # 遍历文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".bin"):  # 确保只处理 .bin 文件
                file_path = os.path.join(folder_path, file_name)
                processed_data = process_srs_file(file_path)
                all_processed_data.append(processed_data)

        # 将所有处理后的数组纵向拼接
        self.data_total = np.vstack(all_processed_data)
          
        valid_len = (self.data_total.shape[0] // (1024)) * 1024
        print("valid_len is", valid_len)
        print("origin_len is", self.data_total.shape[0])

        self.data_total = self.data_total[:valid_len]
        
        # 数据重塑，假设每个输入的特征有32个时间步长和256个样本
        self.data_total = self.data_total.reshape(-1, 1, 1024, self.data_total.shape[1])     
        
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
    folder_path = r'/home/luhan/lap/IndooLocation/Data/srs-38points-1000time-1cycles/01'
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

