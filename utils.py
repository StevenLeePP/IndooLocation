import sys
import numpy as np
import torch
from tqdm import tqdm
import csv

def calculate_accuracy(pred, labels):
    """计算准确率"""
    pred_classes = torch.argmax(pred, dim=1)
    correct = (pred_classes == labels).sum().item()
    total = labels.size(0)
    return correct / total * 100

def train_one_epoch(model, optimizer, data_loader, device, epoch, save_path="train_predictions.csv"):
    model.train()
    loss_function = torch.nn.MSELoss()
    optimizer.zero_grad()
    
    epoch_loss = 0
    epoch_mae = 0  # 初始化平均绝对误差
    num_batches = 0

    # 打开文件以追加预测值和真实标签
    with open(save_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for step, data in enumerate(data_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.squeeze().to(device)
            # 打印 labels 的值
            # print(f"Step {step}: Labels = {labels.cpu().numpy()}")
            pred = model(images)
            # pred = torch.round(pred, decimals=2)  # 将 pred 的值保留到小数点后 2位
            
            try:
                loss = loss_function(pred, labels)
                loss.backward()
                
                # 只写入第一个样本的数据

                # csvwriter.writerow([
                #     step,
                #     pred[0, 0].cpu().detach().numpy(),
                #     pred[0, 1].cpu().detach().numpy(),
                #     labels[0, 0].cpu().detach().numpy(),
                #     labels[0, 1].cpu().detach().numpy()
                # ])

                for i in range(pred.shape[0]):  # 遍历批次中的每个样本
                    csvwriter.writerow([
                        step,
                        pred[i, 0].cpu().detach().numpy(),
                        pred[i, 1].cpu().detach().numpy(),
                        labels[i, 0].cpu().detach().numpy(),
                        labels[i, 1].cpu().detach().numpy()
                    ])
                
                epoch_loss += loss.item()
                
                # 计算平均绝对误差 (MAE)
                mae = torch.mean(torch.abs(pred - labels))
                epoch_mae += mae.item()
                # # 计算均方根误差 (RMSE)
                # rmse = torch.sqrt(torch.mean((pred - labels) ** 2))
                # epoch_rmse += rmse.item()
                
                num_batches += 1
                
                data_loader.desc = f"[train epoch {epoch}] step:{step}, loss: {loss.item():.3f}, MAE: {mae.item():.3f}"
                
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)

                optimizer.step()
                optimizer.zero_grad()
                    
            except RuntimeError as e:
                print(f"Error at step {step}")
                print(f"Labels range: {labels.min().item()} to {labels.max().item()}")
                print(f"Pred shape: {pred.shape}, Labels shape: {labels.shape}")
                raise e

    avg_loss = epoch_loss / num_batches
    avg_mae = epoch_mae / num_batches  # 计算平均 MAE
    print(" ")
    print("[train epoch {}] 平均误差:{:.3f}, 平均绝对误差:{:.3f}".format(
        epoch+1, avg_loss, avg_mae))
    print("====================")

    return avg_loss, avg_mae

@torch.no_grad()
def test_model(model, data_loader, device):
    loss_function = torch.nn.MSELoss()
    model.eval()

    all_preds = []
    all_labels = []
    all_losses = []
    total_accuracy = 0
    num_batches = 0

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.squeeze().to(device)
        pred = model(images)
        loss = loss_function(pred, labels)
        
        # accuracy = calculate_accuracy(pred, labels)
        # total_accuracy += accuracy
        num_batches += 1
        
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())  
        all_losses.append(loss.item())

    # avg_accuracy = total_accuracy / num_batches
    avg_accuracy=0
    pred_all = np.concatenate(all_preds, axis=0)
    labels_all = np.array(all_labels)  
    loss_all = np.array(all_losses)

    return pred_all, loss_all, labels_all, avg_accuracy