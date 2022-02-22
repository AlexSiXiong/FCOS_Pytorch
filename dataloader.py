import os
import pprint

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

"""
Banana Data resource:
http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip

Ped Data resource:
https://www.cis.upenn.edu/~jshi/ped_html/
"""


def read_banana(is_train=True, root_dir='data/banana/'):
    file_name = os.path.join(root_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(file_name)
    csv_data = csv_data.set_index('image_name')

    images, label_position = [], []

    for image_name, label, position in csv_data.iterrows:
        img = Image.open(
            os.path.join(root_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{image_name}')).convert(
            'RGB')
        images.append(img)

        label_position.append([label, position])

    return images, torch.tensor(label_position).unsqueeze(1) / 256


class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, root_dir='data/banana/') -> None:
        self.images, self.label_position = read_banana(is_train, root_dir)
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return self.images[idx].float(), self.label_position[idx]

    def __len__(self):
        return len(self.images)


def read_campus_pedestrian(root_dir='./PennFudanPed2/'):
    annotations = list(sorted(os.listdir(os.path.join(root_dir, "Annotation"))))
    annotation_files = [os.path.join(root_dir, 'Annotation', i) for i in annotations]

    annotation_list = []
    for i in annotation_files:
        annotation_dict = {}
        boxes = []
        with open(i, 'r') as f:
            width = 0
            height = 0

            for line in f:
                if 'Image size' in line:
                    size = line[line.find(':') + 2:]

                    width = int(size[:size.find('x') - 1])
                    height = int(size[size.find('x') + 2:size.rfind('x')])

                    annotation_dict['width'] = width
                    annotation_dict['height'] = height
                if 'Objects' in line:
                    annotation_dict['numbers'] = int(line[line.find(':') + 2: line.find('{') - 1])

                if 'Bounding box' in line:
                    line = line[line.find(':') + 2:]
                    min_coordinates = line[line.find('-') + 2:]
                    max_coordinates = line[:line.find('-') - 1]

                    xmin = int(min_coordinates[min_coordinates.find('(') + 1:min_coordinates.find(',')]) / width
                    ymin = int(min_coordinates[min_coordinates.find(',') + 1:min_coordinates.find(')')]) / height

                    xmax = int(max_coordinates[max_coordinates.find('(') + 1:max_coordinates.find(',')]) / width
                    ymax = int(max_coordinates[max_coordinates.find(',') + 1:max_coordinates.find(')')]) / height
                    boxes.append([xmin, ymin, xmax, ymax])

                annotation_dict['boxes'] = boxes

            annotation_dict['labels'] = torch.ones((annotation_dict['numbers'],), dtype=torch.int64)
            annotation_dict['boxes'] = torch.as_tensor(annotation_dict['boxes'], dtype=torch.float32)
            annotation_dict['numbers'] = torch.as_tensor(annotation_dict['numbers'], dtype=torch.int64)
            annotation_list.append(annotation_dict)
    people_images = list(sorted(os.listdir(os.path.join(root_dir, "PNGImages"))))
    images = [Image.open(os.path.join(root_dir, "PNGImages", i)).convert('RGB') for i in people_images]
    # images[0].show()
    pprint.pprint(annotation_list)
    return images, annotation_list


class CampusPedestrianDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_classes=1) -> None:
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
        """
        self.images, self.annotation_dict = read_campus_pedestrian(root_dir='./PennFudanPed2/')
        self.num_classes = num_classes

    def __getitem__(self, idx):
        return self.images[idx].float(), self.annotation_dict[idx]

    def __len__(self) -> int:
        return len(self.images)


if __name__ == '__main__':
    read_campus_pedestrian()
