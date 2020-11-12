from torch.optim import RMSprop
from torch import optim
import torch
from torch.cuda import amp

from torch.nn import CrossEntropyLoss
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader

import Models as Models
import Dataset as Dataset

from SAM import SAM
from efficientnet_pytorch import EfficientNet

import time
import numpy as np

if __name__ == '__main__':
    # ----- Model ----- #
    LOAD = False
    if LOAD:
        model = torch.load('checkpoints_eff_sam/checkpoint_%04d.pth' % 145).cuda()
    else:
        # model = Models.DN_Classifier().cuda()
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=196).cuda()

    # ----- Data Loader ----- #
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop((300, 300), (0.8, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    preprocess_val = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    val_index = np.random.choice(11185, 1185, replace=False)
    val_mask = np.zeros(11185, dtype=np.bool)
    val_mask[val_index] = True
    train_mask = np.logical_not(val_mask)
    # train_mask = val_mask
    DATA_ROOT = 'E:/Datasets/Homeworks/cs-t0828-2020-hw1/'
    dataset = Dataset.T0828_HW1_Train(DATA_ROOT, preprocess, train_mask)
    dataset_val = Dataset.T0828_HW1_Train(DATA_ROOT, preprocess_val, val_mask)
    data_loader = DataLoader(dataset, batch_size=16, num_workers=2, drop_last=True, shuffle=True)
    val_data_loader = DataLoader(dataset_val, batch_size=16, num_workers=2, drop_last=True, shuffle=True)

    # ----- Optimizer ----- #
    # optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
    base_optimizer = torch.optim.RMSprop
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.00001)
    loss_function = CrossEntropyLoss()
    scaler = amp.GradScaler()
    p = list(model.parameters())[-2]
    # ---- Training Loop ---- #
    t1 = time.process_time()

    best_acc = 0.2

    for epoch in range(5000):
        counter = 0
        for input_img, gt_label in data_loader:
            '''
            optim.zero_grad()

            with amp.autocast():
                pred_labels = model(input_img.cuda())
                loss = loss_function(pred_labels, gt_label.cuda())  # shape: ((N, C), N)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            '''
            # optimizer.zero_grad()
            pred_scores = model(input_img.cuda())
            loss = loss_function(pred_scores, gt_label.cuda())  # shape: ((N, C), N)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            # optimizer.step()

            pred_scores = model(input_img.cuda())
            loss = loss_function(pred_scores, gt_label.cuda())  # shape: ((N, C), N)
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
            if counter % 100 == 0:
                t2 = time.process_time()
                pred_label = torch.argmax(pred_scores.detach(), dim=1).cpu()
                acc = pred_label.eq(gt_label).sum() / 16.0
                print(epoch, counter, 'time', t2 - t1, 'loss:', loss.detach().cpu(), acc)
                t1 = t2
                # print(gt_label.detach().cpu())
                # print(torch.argmax(pred_labels.detach(), dim=1).cpu())
            counter += 1
        correct_count = 0
        loss_sum = 0
        model.eval()
        for val_img, val_label in val_data_loader:
            val_img = val_img.cuda()
            val_label = val_label.cuda()
            pred_scores = model(val_img).detach()
            pred_label = torch.argmax(pred_scores, dim=1)
            correct_count += torch.sum(pred_label == val_label)
            loss_sum += loss_function(pred_scores, val_label.cuda()).cpu()
        
        acc = correct_count / 1185.0
        print('val', epoch, 'acc', acc.detach().cpu(), 'best', best_acc, 'loss_sum', loss_sum)
        model.train()

        if acc > best_acc:
            print('***** save model *****')
            best_acc = acc
            torch.save(model, 'checkpoints/checkpoint_%04d.pth'%epoch)

