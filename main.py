from model.my_transforms import *
from model.model import MLFP
from pre_process.data import MLFPDataset
from torch.utils.data import DataLoader
from utils import *

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLFP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(10)):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10}")
                running_loss = 0.0
