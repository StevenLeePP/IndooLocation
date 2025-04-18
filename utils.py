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

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    
    epoch_loss = 0
    total_accuracy = 0
    num_batches = 0

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.long().squeeze().to(device)
        
        pred = model(images)
        
        try:
            loss = loss_function(pred, labels)
            loss.backward()
            
            accuracy = calculate_accuracy(pred, labels)
            total_accuracy += accuracy
            epoch_loss += loss.item()
            num_batches += 1
            
            data_loader.desc = f"[train epoch {epoch}] step:{step}, loss: {loss.item():.3f}, accuracy: {accuracy:.2f}%"
            
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
    avg_accuracy = total_accuracy / num_batches
    
    print(" ")
    print("[train epoch {}] 平均误差:{:.3f}, 平均准确率:{:.2f}%".format(
        epoch+1, avg_loss, avg_accuracy))
    print("====================")

    return avg_loss, avg_accuracy

@torch.no_grad()
def test_model(model, data_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    all_preds = []
    all_labels = []
    all_losses = []
    total_accuracy = 0
    num_batches = 0

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.long().squeeze().to(device)
        
        pred = model(images)
        loss = loss_function(pred, labels)
        
        accuracy = calculate_accuracy(pred, labels)
        total_accuracy += accuracy
        num_batches += 1
        
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())  
        all_losses.append(loss.item())

    avg_accuracy = total_accuracy / num_batches
    pred_all = np.concatenate(all_preds, axis=0)
    labels_all = np.array(all_labels)  
    loss_all = np.array(all_losses)

    return pred_all, loss_all, labels_all, avg_accuracy