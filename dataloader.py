from logging import root
from torch.utils.data import DataLoader
import pandas as pd
import os, io


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
        img_name = os.path.join(self.root_dir, self.object_frame.iloc[idx, 0])
        image = io.imread(img_name)
