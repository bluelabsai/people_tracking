from pathlib import Path
import base64
import datetime

import cv2
import numpy as np
from imutils.video import FPS
import dlib

from otracking.yolo import YOLO, Yolov5
from config.config import MODELS_DIR, STORE_DIR, DATA_DIR
from otracking.trackingeng import centroidtracker, trackableobject
from otracking.utils import rel2abs_points, poligonal_mask

ALLOWED_DETECTORS = ["yolov3", "ssd_mobilenet", "yv5_onnx", "yv5_pt"]

class PeopleAnalytics:

    def __init__(
        self, camera_location:str, period_time:str, detector_name:str="yolov3", confidence: float=0.6) -> None:
        
        if detector_name not in ALLOWED_DETECTORS:
            raise ValueError(f"detector name not implement try someone: {ALLOWED_DETECTORS}")
        
        self.camera_location = camera_location
        self.period_time = period_time
        self.PATH_VIDEO = "/tmp/video_in.mp4"
        self.PATH_OUTPUT = "/tmp/output.mp4"

        self.confidence = confidence
        if detector_name == "yolov3":
            raise ValueError("yolov3 is not implemented in this branch")
            # self.model_dir = Path(STORE_DIR, "models", "yolov3")
            # self.model = YOLO("yolov3")

        elif detector_name in [ "yv5_onnx", "yv5_pt"]:
            self.model = Yolov5(detector_name, self.confidence)

        elif detector_name == "ssd_mobilenet":
            raise ValueError(f"detector name not implement try someone: {ALLOWED_DETECTORS}")
        


    def process_video(
        self, video_bytes, region_mask=None, draw_video:bool=False):

        self.region_mask = region_mask
        if draw_video:
            self.output = DATA_DIR / "output_video.mp4"
            return self._process_video_show(video_bytes)
        else:
            return self._process_video(video_bytes)

    def _process_video(self, video_bytes):
        
        video_result = open(self.PATH_VIDEO, "wb")
        video_result.write(video_bytes)

        skip_fps = 30

        vs = cv2.VideoCapture(self.PATH_VIDEO)

        # Definimos ancho y alto
        W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.region_mask:
            self.region_mask = [self.region_mask]
            abs_points = rel2abs_points(W, H, self.region_mask)

        ct = centroidtracker.CentroidTracker(maxDisappeared= 40, maxDistance = 50)

        # Inicializamos variables principales
        trackers = []
        trackableObjects = {}
        raw_data = {}

        totalFrame = 0
        totalDown = 0
        totalUp = 0

        DIRECTION_PEOPLE = False

        # Creamos un umbral para sabre si el carro paso de izquierda a derecha o viceversa
        # En este caso lo deje fijo pero se pudiese configurar seg??n la ubicaci??n de la c??mara.
        POINT = [0, int((H/2)-H*0.1), W, int(H*0.1)]

        # Los FPS nos van a permitir ver el rendimiento de nuestro modelo y si funciona en tiempo real.
        fps = FPS().start()

        # Bucle que recorre todo el video
        while True:
            # Leemos el primer frame
            ret, frame = vs.read()

            # Si ya no hay m??s frame, significa que el video termino y por tanto se sale del bucle
            if frame is None:
                break
            
            if self.region_mask:
                frame_input = poligonal_mask(frame, W, H, abs_points)
            else:
                frame_input = frame.copy()

            status = "Waiting"
            rects = []

            date_time = datetime.datetime.now()

            # Nos saltamos los frames especificados.
            if totalFrame % skip_fps == 0:
                status = "Detecting"
                trackers = []
                detections_crop, _ = self.model.predict(frame_input)

                # transorm boxes

                # Recorremos las detecciones
                # for x in range(len(classes)):
                for detection in detections_crop:
                    idx = int(detection["cls"])
                    # Tomamos los bounding box 
                    (startX, startY, endX, endY) = np.array(detection["box"]).astype("int")

                    # Con la funci??n de dlib empezamos a hacer seguimiento de los boudiung box obtenidos
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(frame_input, rect)

                    trackers.append(tracker)
            else:
                # En caso de que no hagamos detecci??n haremos seguimiento
                # Recorremos los objetos que se les est?? realizando seguimiento
                for tracker in trackers:
                    status = "Tracking"
                    # Actualizamos y buscamos los nuevos bounding box
                    tracker.update(frame_input)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    rects.append((startX, startY, endX, endY))

            objects = ct.update(rects)
            objects_to_save = {}

            # Recorremos cada una de las detecciones
            for (objectID, centroid) in objects.items():
                # Revisamos si el objeto ya se ha contado
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = trackableobject.TrackableObject(objectID, centroid)

                else:
                    # Si no se ha contado, analizamos la direcci??n del objeto
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)
                    if not to.counted:
                        if centroid[0] > POINT[0] and centroid[0] < (POINT[0]+ POINT[2]) and centroid[1] > POINT[1] and centroid[1] < (POINT[1]+POINT[3]):
                            if DIRECTION_PEOPLE:
                                if direction >0:
                                    totalUp += 1
                                    to.counted = True
                                else:
                                    totalDown +=1
                                    to.counted = True
                            else:
                                if direction <0:
                                    totalUp += 1
                                    to.counted = True
                                else:
                                    totalDown +=1
                                    to.counted = True

                trackableObjects[objectID] = to
                objects_to_save[objectID] = centroid.tolist()

            if objects_to_save:
                raw_data[str(date_time)] = objects_to_save

            totalFrame += 1
            fps.update()

        # Terminamos de analizar FPS y mostramos resultados finales
        fps.stop()

        print("Tiempo completo {}".format(fps.elapsed()))
        print("Tiempo aproximado por frame {}".format(fps.fps()))

        # Cerramos el stream de consumir el video.
        vs.release()

        output_data = {
        "camera_location": self.camera_location,
        "period_time": self.period_time,
        "raw_data": raw_data
        }

        return {"output_data": output_data, "draw_video":""}

    def _process_video_show(self, video_bytes):
        
        video_result = open(self.PATH_VIDEO, "wb")
        video_result.write(video_bytes)

        skip_fps = 15
        threshold = 0.3

        vs = cv2.VideoCapture(self.PATH_VIDEO)

        writer = None

        # Definimos ancho y alto
        W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.region_mask:
            self.region_mask = [self.region_mask]
            abs_points = rel2abs_points(W, H, self.region_mask)

        ct = centroidtracker.CentroidTracker(maxDisappeared= 40, maxDistance = 50)

        # Inicializamos variables principales
        trackers = []
        trackableObjects = {}
        raw_data = {}

        totalFrame = 0
        totalDown = 0
        totalUp = 0

        DIRECTION_PEOPLE = True

        # Creamos un umbral para sabre si el carro paso de izquierda a derecha o viceversa
        # En este caso lo deje fijo pero se pudiese configurar seg??n la ubicaci??n de la c??mara.
        POINT = [0, int((H*0.7)-H*0.1), W, int(H*0.1)]

        # Los FPS nos van a permitir ver el rendimiento de nuestro modelo y si funciona en tiempo real.
        fps = FPS().start()

        # Definimos el formato del archivo resultante y las rutas.
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(str(self.output), fourcc, 30.0, (W, H), True)
        

        # Bucle que recorre todo el video
        while True:
            # Leemos el primer frame
            ret, frame = vs.read()

            # Si ya no hay m??s frame, significa que el video termino y por tanto se sale del bucle
            if frame is None:
                break

            if self.region_mask:
                frame_input = poligonal_mask(frame, W, H, abs_points)
            else:
                frame_input = frame.copy()

            status = "Waiting"
            rects = []

            date_time = datetime.datetime.now()

            # Nos saltamos los frames especificados.
            if totalFrame % skip_fps == 0:
                status = "Detecting"
                trackers = []
                detections_crop, _ = self.model.predict(frame_input)

                # transorm boxes

                # Recorremos las detecciones
                # for x in range(len(classes)):
                for detection in detections_crop:
                    idx = int(detection["cls"])
                    # Tomamos los bounding box 
                    (startX, startY, endX, endY) = np.array(detection["box"]).astype("int")

                    # Con la funci??n de dlib empezamos a hacer seguimiento de los boudiung box obtenidos
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(frame_input, rect)

                    trackers.append(tracker)
            else:
                # En caso de que no hagamos detecci??n haremos seguimiento
                # Recorremos los objetos que se les est?? realizando seguimiento
                for tracker in trackers:
                    status = "Tracking"
                    # Actualizamos y buscamos los nuevos bounding box
                    tracker.update(frame_input)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    rects.append((startX, startY, endX, endY))

            # Dibujamos el umbral de conteo
            #cv2.rectangle(frame, (POINT[0], POINT[1]), (POINT[0]+ POINT[2], POINT[1] + POINT[3]), (255, 0, 255), 2)

            objects = ct.update(rects)
            objects_to_save = {}

            # Recorremos cada una de las detecciones
            for (objectID, centroid) in objects.items():
                # Revisamos si el objeto ya se ha contado
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = trackableobject.TrackableObject(objectID, centroid)

                else:
                    # Si no se ha contado, analizamos la direcci??n del objeto
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)
                    if not to.counted:
                        if centroid[0] > POINT[0] and centroid[0] < (POINT[0]+ POINT[2]) and centroid[1] > POINT[1] and centroid[1] < (POINT[1]+POINT[3]):
                            if DIRECTION_PEOPLE:
                                if direction >0:
                                    totalUp += 1
                                    to.counted = True
                                else:
                                    totalDown +=1
                                    to.counted = True
                            else:
                                if direction <0:
                                    totalUp += 1
                                    to.counted = True
                                else:
                                    totalDown +=1
                                    to.counted = True

                trackableObjects[objectID] = to
                objects_to_save[objectID] = centroid.tolist()

                # Dibujamos el centroide y el ID de la detecci??n encontrada
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

            if totalFrame % skip_fps == 0:
                if objects_to_save:
                    raw_data[str(date_time)] = objects_to_save
            # Totalizamos los resultados finales
            info = [
                    ("Sur", totalUp),
                    ("Norte", totalDown),
                    ("Estado", status),
            ]

            for (i, (k,v)) in enumerate(info):
              text = "{}: {}".format(k,v)
              cv2.putText(frame, text, (10, H - ((i*20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Almacenamos el framme en nuestro video resultante.
            writer.write(frame)
            totalFrame += 1
            fps.update()

        # Terminamos de analizar FPS y mostramos resultados finales
        fps.stop()

        print("Tiempo completo {}".format(fps.elapsed()))
        print("Tiempo aproximado por frame {}".format(fps.fps()))

        # Cerramos el stream the almacenar video y de consumir el video.
        writer.release()
        vs.release()
        
        video = open(self.output, "rb")
        video_read = video.read()
        image_64_encode = base64.b64encode(video_read)
        image_64_encode_return = image_64_encode.decode() 

        output_data = {
        "camera_location": self.camera_location,
        "period_time": self.period_time,
        "raw_data": raw_data
        }

        return {"output_data": output_data, "draw_video":image_64_encode_return}

 