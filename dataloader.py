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
class FCOSDataset(torch.utils.data.Dataset):
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

