import os
import json
from PIL import Image


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


def read_imgs(frame_index, frame_dir):
    frames = []
    for i in frame_index:
        # find if i has suffix
        if i.find('.') == -1:
            # try possible img suffixes
            for suffix in ['jpg', 'png', 'jpeg']:
                frame_path = f"{frame_dir}/{i}.{suffix}"
                if os.path.exists(frame_path):
                    break
        else:
            frame_path = f"{frame_dir}/{i}"
        frames.append(Image.open(frame_path))
    return frames


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
