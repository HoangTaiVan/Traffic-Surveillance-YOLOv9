import numpy as np
import cv2
import torch
from deep_sort_realtime import deep_sort
from deep_sort_realtime.deepsort_tracker import DeepSort
from pyparsing import White

from models.common import DetectMultiBackend,AutoShape

#config value
video_path=r"D:\deep learning\YOLOv9\yolov9\data_ext\16849-276488425_small.mp4"

conf_threshold=0.5
tracking_class = None

#Khoi tao deep_sort
tracker =DeepSort(max_age=10)

#Khoi tao yolov9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model= DetectMultiBackend(weights=r"D:\deep learning\YOLOv9\yolov9\weights\yolov9-c-converted.pt", device=device, fuse=True)
model =AutoShape(model)

#load classname tu file classes.names
with open(r"D:\deep learning\YOLOv9\yolov9\data_ext\classes.names") as f:
    class_names= f.read().strip().split("\n")

colors=np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")
tracks=[]

#Khoi tao VideoCapture de doc tu file video
cap = cv2.VideoCapture(video_path)

#Tiến hành đọc từng frame từ video
while True:
    #Read
    ret, frame =cap.read()
    if not ret:
        break
    frame_small =cv2.resize(frame, (960, 540))
    #đưa qua model để detect
    results= model(frame_small)

    detect=[]
    for detect_object in results.pred[0]:
        label, confidence, bbox =detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id=int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue

        else:
            if class_id !=tracking_class or confidence < conf_threshold:
                continue

        detect.append(([x1, y1, x2-x1, y2-y1], confidence, class_id))

    #Update, gan ID bang deepsort
    tracks= tracker.update_tracks(detect, frame=frame)

    #Ve len man hinh cac khung chu nhat kem id
    for track in tracks:
        if track.is_confirmed():
            track_id =track.track_id

            ltrb =track.to_ltrb()
            class_id =track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = "{}-{}".format(class_names[class_id], track_id)
            cv2.rectangle(frame_small, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame_small, (x1-1, y1-20), (x1 + len(label)*12, y1), (B,G,R), -1 )
            cv2.putText(frame_small, label, (x1 +5, y1 -8), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 2)

    #Show
    cv2.imshow("Obj_tracking", frame_small)

    #Bam Q thi thoat
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()