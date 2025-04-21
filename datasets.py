import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import tqdm
class MyDataSet(Dataset):
    def __init__(self,folder_path):
        # 获取文件夹下所有文件的文件名列表
        all_files = os.listdir(folder_path)
        # 筛选出所有CSV文件
        csv_files = [f for f in all_files if f.endswith('.csv')]
        # CSV文件个数
        csv_num = (len(csv_files))
        data_total =[]
        data_total = np.array(data_total)
        for j in tqdm.tqdm(range(csv_num)):
            temp_path = csv_files[j]
            temp_csv_path = folder_path+"/"+ temp_path
            input_data = np.array(pd.read_csv(temp_csv_path))
            if j == 0:
                data_total = input_data
            else:
                data_total = np.vstack((data_total,input_data))
            #----
        self.data_total = data_total.astype(np.float32)
        # valid_len = (self.data_total.shape[0] // (256*32)) * 256*32
        valid_len = (self.data_total.shape[0] // (256*32)) * 256*32

        print("valid_len is",valid_len)
        print("origin_len is",self.data_total.shape[0])

        self.data_total = self.data_total[:valid_len]
        
        self.data_total = self.data_total.reshape(-1, 32,256, self.data_total.shape[1])
        self.data_total[:,:,:,0] = self.data_total[:,:,:,0].clip(1, 5)
        self.len = self.data_total.shape[0]
        
        print('')
        print('===============')
        print(' ')
        print('输入数据总维度为 ', self.data_total[:,:,:,1:].shape)  # (N, 256, 8)
        print('label总维度为 ', self.data_total[:,:,:,0].shape)     # (N, 256)
        print(' ')
        print('===============')
        print('')
        
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        features = torch.FloatTensor(self.data_total[index,:, :, 1:])  # (N,32,256, 8)
        label = torch.LongTensor([int(self.data_total[index,0, 0, 0])-1])  # 标签减1，使其范围变为0-38
        label = label.squeeze() 
        return features, label

