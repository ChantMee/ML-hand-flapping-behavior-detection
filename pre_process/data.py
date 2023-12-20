from utils import *
import random
import os

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MLFPDataset(Dataset):
    """
    duration: how long (seconds) should one data be
    num_frame: how many frames should extract from one duration
    """
    def __init__(self, annotation_dir, frame_dir, duration, num_frame, transform=None):
        self.annotation = read_annotation(annotation_dir)
        self.frame_dir = frame_dir
        self.duration = duration
        self.num_frame = num_frame
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        video_id = self.annotation[idx]['video_id']
        start_time = self.annotation[idx]['start_time']
        end_time = self.annotation[idx]['end_time']
        video_class = self.annotation[idx]['video_class']
        fps = int(self.annotation[idx]['fps'])
        frame_list = self.annotation[idx]['frame_list']

        interval = self.duration * fps // self.num_frame
        # random select a start frame
        start_frame_index = random.randint(0, interval - 1)
        # get frame index
        frame_index = [i for i in range(start_frame_index, (end_time - start_time) * fps, interval)]
        if len(frame_index) < self.num_frame:
            frame_index += [frame_index[-1]] * (self.num_frame - len(frame_index))
        # randomly select consecutive self.num_frame frames
        start_frame_index = random.randint(0, len(frame_index) - self.num_frame)
        selected_frame_index = frame_index[start_frame_index: start_frame_index + self.num_frame]
        selected_frame = [frame_list[i] for i in selected_frame_index]
        # read frames
        frames = read_imgs(selected_frame, self.frame_dir)
        frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return frames, video_class


if __name__ == '__main__':
    dataset_path = r'C:\Users\chant\OneDrive\Courses\WOA7015 Advanced Machine Learning\final\dataset'
    video_folder_name = r'videos'
    video_duration_annotation_name = r'video_duration_annotation.csv'
    video_annotation_name = r'video_annotation.json'
    save_folder_name = r'frames'

    # transform with resize, to_tensor, crop, rotate, normalize
    transform = torchvision.transforms.Compose([
        transforms.Resize((128, 171)),
        transforms.RandomCrop((112, 112)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

    dataset = MLFPDataset(os.path.join(dataset_path, video_annotation_name),
                          os.path.join(dataset_path, save_folder_name),
                          duration=2,
                          num_frame=16,
                          transform=transform,
                          )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    imgs, labels = next(iter(dataloader))
    print(imgs.shape, labels.shape)
