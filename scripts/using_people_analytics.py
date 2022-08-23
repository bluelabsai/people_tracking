import cv2
from pathlib import Path

from otracking.models import PeopleAnalytics
from config.config import DATA_DIR

name_file = "4_seccion_tienda.mp4"
path_video = DATA_DIR / name_file
path_out = DATA_DIR / ("out_" + name_file)

camera_location = "pasillo_2"
period_time = "2022-08-05-10am_2022-08-05-11am"

skip_fps = 30
threshold = 0.3

contents = cv2.VideoCapture(str(path_video))

with open(path_video, "rb") as file:
    contents = file.read()

model = PeopleAnalytics(camera_location, period_time, detector_name="yv5_onnx")
response = model.process_video(contents, True, path_out, portion_mask=[0, 0.3, 1, 0.7])

video_bs64 = response["draw_video"]
