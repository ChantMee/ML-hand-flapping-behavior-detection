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
    return frame_list


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


def read_annotation(annotation_path):
    """
    Read annotation file
    :param annotation_path: path to annotation file
    :return: dict of annotation
    """
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
        f.close()
    return annotation

def write_annotation(annotation_path, annotation):
    """
    Write annotation file
    :param annotation_path: path to annotation file
    :param annotation: dict of annotation
    :return: None
    """
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f)
        f.close()


if __name__ == "__main__":
    dataset_path = r'C:\Users\chant\OneDrive\Courses\WOA7015 Advanced Machine Learning\final\dataset'
    video_folder_name = r'videos'
    video_duration_annotation_name = r'video_duration_annotation.csv'
    video_annotation_name = r'video_annotation.json'
    save_folder_name = r'frames'

    rda = read_duration_annotation(os.path.join(dataset_path, video_duration_annotation_name))
    annotation = {0: [], 1: []}
    for i in tqdm(range(len(rda))):
        video_id = rda[i][0]
        start_time, end_time = rda[i][1]
        video_class = rda[i][2]
        video_path = os.path.join(dataset_path, video_folder_name, str(video_id) + '.mp4')
        save_folder_path = os.path.join(dataset_path, save_folder_name)
        frame_list = extract_frame(video_id, video_path, start_time, end_time, save_folder_path)
        annotation[video_class].append(frame_list)
        write_annotation(os.path.join(dataset_path, video_annotation_name), annotation)
