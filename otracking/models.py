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
from otracking.utils import relativebox2absolutebox

ALLOWED_DETECTORS = ["yolov3", "ssd_mobilenet", "yv5_onnx", "yv5_pt"]

class PeopleAnalytics:

    def __init__(self, camera_location:str, period_time:str, detector_name:str="yolov3") -> None:
        
        if detector_name not in ALLOWED_DETECTORS:
            raise ValueError(f"detector name not implement try someone: {ALLOWED_DETECTORS}")
        
        self.camera_location = camera_location
        self.period_time = period_time
        self.PATH_VIDEO = "/tmp/video_in.mp4"
        self.PATH_OUTPUT = "/tmp/output.mp4"

        if detector_name == "yolov3":
            raise ValueError("yolov3 is not implemented in this branch")
            # self.model_dir = Path(STORE_DIR, "models", "yolov3")
            # self.model = YOLO("yolov3")

        elif detector_name in [ "yv5_onnx", "yv5_pt"]:
            self.model = Yolov5(detector_name, 0.3)

        elif detector_name == "ssd_mobilenet":
            raise ValueError(f"detector name not implement try someone: {ALLOWED_DETECTORS}")
        


    def process_video(
        self, video_bytes, draw_video:bool=False, output = DATA_DIR / "output_video.mp4", portion_mask=None):
        
        self.portion_mask = portion_mask
        if draw_video:
            self.output = output
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
        # En este caso lo deje fijo pero se pudiese configurar según la ubicación de la cámara.
        POINT = [0, int((H/2)-H*0.1), W, int(H*0.1)]

        # Los FPS nos van a permitir ver el rendimiento de nuestro modelo y si funciona en tiempo real.
        fps = FPS().start()

        # Bucle que recorre todo el video
        while True:
            # Leemos el primer frame
            ret, frame = vs.read()

            # Si ya no hay más frame, significa que el video termino y por tanto se sale del bucle
            if frame is None:
                break
            
            status = "Waiting"
            rects = []

            date_time = datetime.datetime.now()

            # Nos saltamos los frames especificados.
            if totalFrame % skip_fps == 0:
                status = "Detecting"
                trackers = []
                # Tomamos la imagen la convertimos a array luego a tensor
                image_np = np.array(frame)

                # pimage = process_image_yolo(image_np)

                # Predecimos los objectos y clases de la imagen
                #boxes, classes, scores = self.model.predict(image_np)
                detections_crop, _ = self.model.predict(image_np)

                # transorm boxes

                # Recorremos las detecciones
                # for x in range(len(classes)):
                for detection in detections_crop:
                    idx = int(detection["cls"])
                    # Tomamos los bounding box 
                    (startX, startY, endX, endY) = np.array(detection["box"]).astype("int")

                    # Con la función de dlib empezamos a hacer seguimiento de los boudiung box obtenidos
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(frame, rect)

                    trackers.append(tracker)
            else:
                # En caso de que no hagamos detección haremos seguimiento
                # Recorremos los objetos que se les está realizando seguimiento
                for tracker in trackers:
                    status = "Tracking"
                    # Actualizamos y buscamos los nuevos bounding box
                    tracker.update(frame)
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
                    # Si no se ha contado, analizamos la dirección del objeto
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
        # En este caso lo deje fijo pero se pudiese configurar según la ubicación de la cámara.
        POINT = [0, int((H*0.7)-H*0.1), W, int(H*0.1)]

        # Los FPS nos van a permitir ver el rendimiento de nuestro modelo y si funciona en tiempo real.
        fps = FPS().start()

        # Definimos el formato del archivo resultante y las rutas.
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(str(self.output), fourcc, 30.0, (W, H), True)

        #mask
        mask = np.zeros((H, W), dtype="uint8")
        if self.portion_mask:
            xy_1 = (int(self.portion_mask[0]*W), int(self.portion_mask[1]*H))
            xy_2 = (int(self.portion_mask[2]*W), int(self.portion_mask[3]*H))
            cv2.rectangle(mask, xy_1, xy_2, 255, -1)
        

        # Bucle que recorre todo el video
        while True:
            # Leemos el primer frame
            ret, frame = vs.read()

            # Si ya no hay más frame, significa que el video termino y por tanto se sale del bucle
            if frame is None:
                break

            if self.portion_mask:
                frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

            status = "Waiting"
            rects = []

            date_time = datetime.datetime.now()

            # Nos saltamos los frames especificados.
            if totalFrame % skip_fps == 0:
                status = "Detecting"
                trackers = []
                # Tomamos la imagen la convertimos a array luego a tensor
                image_np = np.array(frame_masked)

                # pimage = process_image_yolo(image_np)

                # Predecimos los objectos y clases de la imagen
                #boxes, classes, scores = self.model.predict(image_np)
                detections_crop, _ = self.model.predict(image_np)

                # transorm boxes

                # Recorremos las detecciones
                # for x in range(len(classes)):
                for detection in detections_crop:
                    idx = int(detection["cls"])
                    # Tomamos los bounding box 
                    (startX, startY, endX, endY) = np.array(detection["box"]).astype("int")

                    # Con la función de dlib empezamos a hacer seguimiento de los boudiung box obtenidos
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(frame, rect)

                    trackers.append(tracker)
            else:
                # En caso de que no hagamos detección haremos seguimiento
                # Recorremos los objetos que se les está realizando seguimiento
                for tracker in trackers:
                    status = "Tracking"
                    # Actualizamos y buscamos los nuevos bounding box
                    tracker.update(frame_masked)
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
                    # Si no se ha contado, analizamos la dirección del objeto
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

                # Dibujamos el centroide y el ID de la detección encontrada
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

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
        
        video = open(self.PATH_OUTPUT, "rb")
        video_read = video.read()
        image_64_encode = base64.b64encode(video_read)
        image_64_encode_return = image_64_encode.decode() 

        output_data = {
        "camera_location": self.camera_location,
        "period_time": self.period_time,
        "raw_data": raw_data
        }

        return {"output_data": output_data, "draw_video":image_64_encode_return}

 