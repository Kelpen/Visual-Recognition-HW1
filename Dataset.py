import csv
import numpy as np
from PIL import Image

from torch.utils.data.dataset import Dataset


class T0828_HW1_Train(Dataset):
    def __init__(self, data_root, transform, part=None) -> None:
        super().__init__()
        self.data_root = data_root
        self.label_file = self.data_root + 'training_labels.csv'
        self.train_data_dir = self.data_root + 'training_data/training_data/'
        with open(self.label_file) as label_file:
            reader = csv.reader(label_file)
            next(reader)  # Pop out the title: ['id', 'label']

            label_dict = {}  # format {label: {'instance_num': int, 'label_id': int}}
            label_id_counter = 0
            data_list = []
            for img_id, label in reader:  # Each line is ['#imgnum', 'label']
                if label not in label_dict.keys():
                    label_dict[label] = {'instance_num': 0, 'label_id': label_id_counter}
                    label_id_counter += 1
                else:
                    label_dict[label]['instance_num'] += 1
                data_list.append([int(img_id), label_dict[label]['label_id']])
            if part is None:
                self.data_list = np.array(data_list)
            else:
                self.data_list = np.array(data_list)[part]

            self.transform = transform
            self.data_num = self.data_list.shape[0]

            self.id_2_label = list(label_dict.keys())  # map the class id back to string label

    def __getitem__(self, index):
        img_id, label = self.data_list[index]
        img: Image.Image = Image.open(self.train_data_dir + '%06d.jpg' % img_id).convert('RGB')
        img = self.transform(img)
        return img, label.astype(np.int64)

    def __len__(self):
        return self.data_num


class T0828_HW1_Test(Dataset):
    def __init__(self, data_root, transform) -> None:
        super().__init__()
        self.data_root = data_root
        self.list_file_name = self.data_root + 'test_data_list.txt'
        with open(self.list_file_name) as list_file:
            self.test_list = []
            for line in list_file:
                self.test_list.append(line[:-1])
            self.transform = transform
            self.data_num = len(self.test_list)

    def __getitem__(self, index):
        img_path = self.test_list[index]
        img: Image.Image = Image.open(self.data_root + img_path).convert('RGB')
        img = self.transform(img)
        return img, img_path[-10:-4]  # image, image_id(in string)

    def __len__(self):
        return self.data_num
