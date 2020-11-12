import torch

from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader

import Models
import Dataset

import csv

if __name__ == '__main__':
    # ----- Model ----- #
    model = torch.load('checkpoints_eff_sam_b1_rms/checkpoint_%04d.pth' % 25).cuda().eval()
    #model = Models.DN_Classifier()
    # ----- Data Loader ----- #
    preprocess_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # train_mask = val_mask
    DATA_ROOT = 'E:/Datasets/Homeworks/cs-t0828-2020-hw1/'
    dataset = Dataset.T0828_HW1_Train(DATA_ROOT, None)
    dataset_test = Dataset.T0828_HW1_Test(DATA_ROOT, preprocess_test)
    data_loader = DataLoader(dataset_test, batch_size=32, num_workers=8)

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for img, ids in data_loader:
            pred_labels: torch.Tensor = model(img.cuda())
            pred_labels = pred_labels.detach().cpu()

            for label, img_id in zip(pred_labels, ids):
                writer.writerow([img_id, dataset.id_2_label[torch.argmax(label)]])
    # print(label_list)




