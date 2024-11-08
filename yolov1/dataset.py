import os

import torch
import pandas as pd
from PIL import Image



class VOCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_file, img_dir, label_dir,
                 split=7, num_box=2, num_classes=20,
                 transform=None,
                 ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.split = split
        self.num_box = num_box
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Handel the .csv & image files
        label_path = os.path.join(self.label_dir, self.annotations.iloc[idx, 1])
        boxes = []
        with open(label_path, 'r') as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) for x in label.replace("\n", "").split("|")
                ]

            boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path)
        boxes = torch.Tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros(self.split,
                                   self.split,
                                   self.num_classes + 5 * self.num_box)

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.split * y), int(self.split * x)
            x_cell, y_cell = (self.split * x - j,
                              self.split * y - i)
            width_cell, height_cell = (self.split * width,
                                  self.split * height)

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.Tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
