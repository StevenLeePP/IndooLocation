import os
import argparse
import torch
from datasets import MyDataSet
# from vit_model import VisionTransformer
from Residual import Residual
from Residual import Student


import collections
import math
import shutil
import pandas as pd
import numpy as np
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from d2l import torch as d2l
from PIL import Image
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
singlefile_data_num = 400  #每个文件训练读取数据个数
singlefile_val_data_num = 100 #每个文件验证读取数据个数
batch_size = 64
epochs = 101
lr = 0.005    #学习率大小
train_path = r"/home/luhan/lap/IndooLocation/Data/train"
test_path =  r"/home/luhan/lap/IndooLocation/Data/test"
if os.path.exists("./weights") is False:
    os.makedirs("./weights")
# 实例化训练数据集
print("读取训练集数据，每个文件读{}条数据".format(singlefile_data_num))
train_dataset = MyDataSet(folder_path=train_path)
print("读取验证集数据，每个文件读{}条数据".format(singlefile_val_data_num))
val_dataset = MyDataSet(folder_path = test_path)

nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
nw = 0 # in windows
print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,  #weight_decay=1e-3
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw)
                                            # 清空txt数据
with open("loss.txt", "w") as f:
    f.write("")
with open("accuracy.txt", "w") as f:
    f.write("")
with open("accuracy_test.txt", "w") as f:
    f.write("")
with open("loss_test.txt", "w") as f:
    f.write("")

print("total epochs:",epochs)
from utils import train_one_epoch,test_model
print("using ",torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model=Student().to(device)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
for epoch in range(epochs):
    # train
    train_loss, train_accuracy = train_one_epoch(model=model,
                                optimizer=optimizer,
                                data_loader=train_loader,
                                device=device,
                                epoch=epoch)
    optimizer.step()
    with open("loss.txt", "a") as f_loss:
        f_loss.write("loss:{}\n".format(train_loss))
    with open("accuracy.txt", "a") as f_accuracy:
        f_accuracy.write("accuracy:{}\n".format(train_accuracy))
    # validate
    if (epoch+1) % 5 == 0:
        pred, test_loss, labels, test_accuracy = test_model(model=model,
                                data_loader=val_loader,
                                device=device)
        average_loss = sum(test_loss)/len(test_loss)
        print("................")
        print("验证集结果：")
        print(f"平均误差: {average_loss:.3f}")
        print(f"准确率: {test_accuracy:.2f}%")
        print("................")
        with open("accuracy_test.txt", "a") as f:
            f.write("accuracy test:{}\n".format(test_accuracy))
        with open("loss_test.txt", "a") as f:
            f.write("loss test:{}\n".format(average_loss))
        
    if (epoch+1) % 20 == 0:
        print("保存模型")
        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        if not os.path.exists('./model_save'):
            os.makedirs('./model_save')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'accuracy': train_accuracy
            }, "./model_save/model-{}.pth".format(epoch))
print("训练完成")
