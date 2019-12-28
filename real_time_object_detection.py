# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from datetime import datetime,timedelta

"""
CLASSES = ["background", "bolsas_te", "botella_plastica", "botella_vidrio", "carton_alimento",
	"carton_caja", "metal_latas", "papel", "residuo_banano", "residuo_huevo", "residuo_manzana", "residuo_naranaja"]
"""
CLASSES = ["background", "organico", "plastico", "vidrio", "cartòn o papel",
	"cartòn o papel", "metal", "cartòn o papel", "organico", "organico", "organico", "organico"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graphV4.pb", "graph.pbtxt")

print("[INFO] starting video stream...")
#vs = VideoStream(src=1).start()
vs = VideoStream(src=0).start()


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def red_neuronal(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.98:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            roi = frame[startY:endY, startX:endX]
            label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
            print(label)
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
            #y = startY - 15 if startY - 15 > 15 else startY + 15
            #cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            #cv2.putText(roi, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            #cv2.imshow("Resultado", frame)
    print("--------------------------------------------------")
    return frame

tiempo_toma = 3
while True:
    frame = vs.read()
    original = imutils.resize(frame, width=300)
    cv2.imshow("Captura", original)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("w"):
        hora_actual = datetime.now()
        seg_ant = hora_actual.second
        loop_toma = True
        cont_tomas = 0
        while loop_toma:
            frame = vs.read()
            original = imutils.resize(frame, width=300)
            cv2.imshow("Captura", original)
            key = cv2.waitKey(1) & 0xFF
            hora_actual = datetime.now()
            seg_act = hora_actual.second
            if (seg_act != seg_ant):
                tiempo_toma-=1
                seg_ant = seg_act
            if(tiempo_toma == 0):
                cont_tomas+=1
                red_neuronal(original)
                #cv2.imshow("toma"+str(cont_tomas), original)
                tiempo_toma = 3
            if(cont_tomas >= 5):
                loop_toma = False
                
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
