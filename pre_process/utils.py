import os
import json
import re

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


def save_annotation(annotation_path, annotation):
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f)
        f.close()


def read_annotation(annotation_path):
    if not os.path.exists(annotation_path):
        return []
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
        f.close()
    return annotation


def read_imgs(img_index, img_dir):
    imgs = []
    for i in img_index:
        # find if i has suffix
        if '.' not in str(i):
            # try possible img suffixes
            for suffix in ['jpg', 'png', 'jpeg']:
                frame_path = f"{img_dir}\\{i}.{suffix}"
                if os.path.exists(frame_path):
                    break
        else:
            frame_path = f"{img_dir}\\{i}"
        # read img with RGB mode
        if os.path.exists(frame_path):
            imgs.append(Image.open(frame_path).convert('RGB'))
    return imgs


def show_imgs(imgs, n_column=4):
    n = len(imgs)
    n_row = n // n_column
    if n % n_column != 0:
        n_row += 1
    fig, axes = plt.subplots(n_row, n_column, figsize=(n_column * 4, n_row * 4))

    for i in range(n_row):
        for j in range(n_column):
            ax = axes[i, j] if n_row > 1 else axes[j]
            idx = i * n_column + j
            if idx < n:
                img = imgs[idx]
                # 检查是否为PyTorch张量
                if isinstance(img, torch.Tensor):
                    # 调整通道顺序并进行反归一化
                    img = img.numpy().transpose((1, 2, 0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = img * std + mean
                    img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.axis('off')
    plt.show()



def read_duration_annotation(video_duration_annotation_path):
    """
    Read annotation file
    :param video_duration_annotation_path: path to annotation file
    :return: list of annotation
    """
    df = pd.read_csv(video_duration_annotation_path).values.tolist()
    for i in range(len(df)):
        match = re.match(r'(\d+):(\d+)-(\d+):(\d+)', df[i][1])
        if match:
            sm, ss, em, es = map(int, match.groups())
            df[i][1] = (sm * 60 + ss, em * 60 + es)
    return df
