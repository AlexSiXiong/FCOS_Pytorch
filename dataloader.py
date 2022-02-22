from logging import root
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image

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
        img = Image.open(os.path.join(root_dir, 'bananas_train' if is_train else 'bananas_val', 'images', f'{image_name}')).convert('RGB')
        images.append(img)

        label_position.append([label, position])

    return images, torch.tensor(label_position).unsqueeze(1)/256

class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, root_dir='data/banana/') -> None:
        self.images, self.label_position = read_banana(is_train, root_dir)
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.images[idx].float(), self.label_position[idx])

    def __len__(self):
        return len(self.images)


class CampusPedestrianDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, num_classes=10) -> None:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.object_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.num_classes = num_classes
        
        
    def __len__(self) -> int:
        return len()

    def _process_img(self):
        pass

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.object_frame.iloc[idx, 0])
        image = Image.open(image_path).convert('RGB')


        num_objects=len(object_ids)

