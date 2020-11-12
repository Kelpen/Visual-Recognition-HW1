import torch

from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser

import Models
import Dataset

import csv


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_root', dest="data_root", help="Where you can find training_labels.csv")
    parser.add_argument("--load_epoch", dest="load_epoch",
                        help="Which epoch you want to load. Only meaningful when load is True")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    load_epoch = args.load_epoch
    DATA_ROOT = args.data_root

    # ----- Model ----- #
    model = torch.load('checkpoints/checkpoint_%04d.pth' % load_epoch).cuda().eval()

    # ----- Data Loader ----- #
    preprocess_test = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    dataset = Dataset.T0828_HW1_Train(DATA_ROOT, None)
    dataset_test = Dataset.T0828_HW1_Test(DATA_ROOT, preprocess_test)
    data_loader = DataLoader(dataset_test, batch_size=32, num_workers=4)

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for img, ids in data_loader:
            pred_labels: torch.Tensor = model(img.cuda())
            pred_labels = pred_labels.detach().cpu()

            for label, img_id in zip(pred_labels, ids):
                writer.writerow([img_id, dataset.id_2_label[torch.argmax(label)]])
    print('Output is saved to output.csv')
