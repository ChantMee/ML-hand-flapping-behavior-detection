from model.my_transforms import *
from model.model import *
from pre_process.data import MLFPDataset
from torch.utils.data import DataLoader
from utils import *
from model.trainer import train

from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm

# config
dataset_path = r'C:\Users\chant\OneDrive\Courses\WOA7015 Advanced Machine Learning\final\dataset'
video_annotation_name = r'video_annotation.json'
save_folder_name = r'frames'

batch_size = 4
lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    dataset = MLFPDataset(annotation_dir=os.path.join(dataset_path, video_annotation_name),
                            frame_dir=os.path.join(dataset_path, save_folder_name),
                            duration=2,
                            num_frame=16,
                            transform=get_transform(0))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model2().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, device, train_dataloader, test_dataloader, criterion, optimizer, 10, 10)