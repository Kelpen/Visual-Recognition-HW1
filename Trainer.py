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
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_root', dest="data_root", help="Where you can find training_labels.csv")
    parser.add_argument('--load', dest="load", help="Weather to load previously trained models", default='False')
    parser.add_argument("--load_epoch", dest="load_epoch", help="Which epoch you want to load. Only meaningful when load is True")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # ---------- Model ---------- #
    LOAD = (args.load == 'True')
    # DATA_ROOT = 'E:/Datasets/Homeworks/cs-t0828-2020-hw1/'
    DATA_ROOT = args.data_root

    if LOAD:
        start_epoch = int(args.load_epoch)
        model = torch.load('checkpoints/checkpoint_%04d.pth' % start_epoch).cuda()
    else:
        start_epoch = 0
        ### Modify model here ###
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=196).cuda()

    # ---------- Data Loader ---------- #
    # Data Augmentation #
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
    # Training Validation Set #
    val_index = np.random.choice(11185, 1185, replace=False)
    val_mask = np.zeros(11185, dtype=np.bool)
    val_mask[val_index] = True
    train_mask = np.logical_not(val_mask)
    # Data Loader #
    dataset = Dataset.T0828_HW1_Train(DATA_ROOT, preprocess, train_mask)
    dataset_val = Dataset.T0828_HW1_Train(DATA_ROOT, preprocess_val, val_mask)
    data_loader = DataLoader(dataset, batch_size=8, num_workers=2, drop_last=True, shuffle=True)
    val_data_loader = DataLoader(dataset_val, batch_size=8, num_workers=2, drop_last=True, shuffle=True)

    # ---------- Optimizer ---------- #
    base_optimizer = optim.Adam
    # base_optimizer = optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.00001, weight_decay=1.e-6)

    loss_function = CrossEntropyLoss()

    # --------- Training Loop --------- #
    t1 = time.process_time()

    best_acc = 0.2

    for epoch in range(start_epoch, 5000):
        counter = 0
        for input_img, gt_label in data_loader:
            pred_scores = model(input_img.cuda())
            loss = loss_function(pred_scores, gt_label.cuda())  # shape: ((N, C), N)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            pred_scores = model(input_img.cuda())
            loss = loss_function(pred_scores, gt_label.cuda())  # shape: ((N, C), N)
            loss.backward()
            optimizer.second_step(zero_grad=True)
            
            if counter % 100 == 0:  # print current status
                t2 = time.process_time()
                pred_label = torch.argmax(pred_scores.detach(), dim=1).cpu()
                acc = pred_label.eq(gt_label).sum() / 16.0
                print(epoch, counter, 'time', t2 - t1, 'loss:', loss.detach().cpu(), acc)
                t1 = t2
            counter += 1

        # Run on Validation Set
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
            torch.save(model, 'checkpoints/checkpoint_%04d.pth' % epoch)
