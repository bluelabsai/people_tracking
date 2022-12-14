from typing import List
from pathlib import Path

import numpy as np
import cv2
import gdown
from pydantic import BaseModel, Field

from config.config import DATA_DIR, MODELS_DIR, URL_MODELS

def relativebox2absolutebox(box: List) -> List:
    box_ = [box[0] + box[2], box[1] + box[3]]
    return box[:2] + box_

def process_image_yolo(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

def draw(img, boxes, classes, scores, labels, colors):

    for box, cl, score in zip(boxes, classes, scores):
        x_start, y_start, w_end, h_end = box
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[cl]]
        cv2.rectangle(img, (x_start, y_start), (w_end, h_end), color, 6)
        text = "{}: {:.4f}".format(labels[cl], score)
        cv2.putText(img, text, (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, color, 3)

    return img

def read_yolo_labels(model_dir:Path, labels_name: str="coco"):
    labels = open(model_dir / F"{labels_name}.names").read().strip().split("\n")
    return labels

def download_models(model_name: str):
  print("download fold from Google Drive ...")
  file_id = URL_MODELS[model_name]
  destination = Path(MODELS_DIR, model_name)

  gdown.download_folder(
    id=file_id,
     output=str(destination),
     use_cookies=False
)

def rel2abs_points(weigth: int, height:int, relative_points: List[List]) -> np.ndarray:
    absolute_points = []
    for relative_point in relative_points:
        obsolute_point = list(np.multiply([weigth, height], relative_point).astype(np.int32))
        absolute_points.append(obsolute_point)
    return np.array(absolute_points)

def poligonal_mask(img: np.ndarray, weigth:int, height:int, points: np.ndarray):
    mask = np.zeros((height, weigth), np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img