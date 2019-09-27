import cv2
import time
import imutils
from imutils import face_utils
import numpy as np
import dlib
import os
import lrw_network as network
import cnn
import tensorflow as tf
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#input_avi = 'lhw_1.mp4'
input_avi = 'input_avi/demo_new_pretrained.mp4'
cap = cv2.VideoCapture(input_avi)  # for stored file
ret, frame= cap.read()
frame= imutils.resize(frame, height=480)
shape_predictor = 'weight/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# To find width/height size
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('{}_lip.avi'.format(input_avi[:-4]),fourcc, 30.0,(110,110))
h_len=0
a=[]
b=[]
c=[]
while(cap.isOpened()):
    # start frame
    ret, frame = cap.read()
    if ret==False:
        break;
    frame = imutils.resize(frame, height=480)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects): 
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        if h_len==0:
            h_len = int(0.55*(shape[54][0]-shape[48][0]))
        center = [int(0.5*(shape[62][1]+shape[66][1])), int(0.5*(shape[62][0]+shape[66][0]))]
    lip_frame = gray[center[0]-h_len:center[0]+h_len,center[1]-h_len:center[1]+h_len]
    a.append((lip_frame<30).sum())
    b.append((lip_frame<20).sum())
    c.append((lip_frame<10).sum())
    #lip_frame = imutils.resize(lip_frame, width=110)

    out.write(lip_frame)
    cv2.rectangle(frame, (center[1]-h_len, center[0]-h_len), (center[1]+h_len, center[0]+h_len), (0, 0, 255), 2)
    cv2.imshow("frame",lip_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
print(a)
print(b)
print(c)
cap.release()
cv2.destroyAllWindows()
out.release()