from utils import *

import cv2
import os
import pandas as pd
import re
import json
from tqdm import tqdm


def extract_frame(video_id, video_path, start_time, end_time, save_path):
    """
    Extract frames from video
    :param video_id: id of video
    :param video_path: path to video
    :param start_time: start time of video
    :param end_time: end time of video
    :param save_path: path to save frames
    :return: None
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    current_frame = start_frame

    frame_list = []
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{save_path}/{video_id}_{current_frame}.jpg", frame)
        current_frame += 1
        frame_list.append(f"{video_id}_{current_frame}.jpg")

    cap.release()
    return frame_list, fps





if __name__ == "__main__":
    dataset_path = r'C:\Users\chant\OneDrive\Courses\WOA7015 Advanced Machine Learning\final\dataset'
    video_folder_name = r'videos'
    video_duration_annotation_name = r'video_duration_annotation.csv'
    video_annotation_name = r'video_annotation.json'
    save_folder_name = r'frames'

    rda = read_duration_annotation(os.path.join(dataset_path, video_duration_annotation_name))
    annotation = read_annotation(os.path.join(dataset_path, video_annotation_name))
    for i in tqdm(range(len(rda))):
        video_id = rda[i][0]
        start_time, end_time = rda[i][1]
        video_class = rda[i][2]
        video_path = os.path.join(dataset_path, video_folder_name, str(video_id) + '.mp4')
        save_folder_path = os.path.join(dataset_path, save_folder_name)
        frame_list, fps = extract_frame(video_id, video_path, start_time, end_time, save_folder_path)
        anno = {
            'video_id': video_id,
            'start_time': start_time,
            'end_time': end_time,
            'video_class': video_class,
            'frame_list': frame_list,
            'fps': fps
        }
        annotation.append(anno)
        save_annotation(os.path.join(dataset_path, video_annotation_name), annotation)
