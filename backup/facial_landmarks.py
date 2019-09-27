# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import time

shape_predictor = 'weight/shape_predictor_68_face_landmarks.dat'
print('1st Stage: Crop Face')
cap=cv2.VideoCapture('output.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi',fourcc, 25.0,(500,500))

# For frame to be fixed by 29
frlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frperiod = int(frlength/29)
if frlength % 2 ==0:
	frame29= range(int(frlength/2-(14*frperiod)),int(frlength/2+(14*frperiod)+1),frperiod)
else:
	frame29= range(int((frlength+1)/2-(14*frperiod)),int((frlength+1)/2+(14*frperiod)+1),frperiod)
count=0

# angle list measuring default
angle=[]
h2=[]
while(cap.isOpened()):
	count=count+1
	#print('count:',count)

	ret, frame = cap.read()
	if ret==False:
	    break;
	if count in frame29:
	    image = frame	
	    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
	    detector = dlib.get_frontal_face_detector()
	    predictor = dlib.shape_predictor(shape_predictor)
	    # load the input image, resize it, and convert it to grayscale    #image = cv2.imread(image)
	    #image = imutils.resize(image, width=500)
	    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	    # detect faces in the grayscale image
	    rects = detector(gray, 1)
	    # loop over the face detections
	    for (i, rect) in enumerate(rects):
	        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
	        shape = predictor(gray, rect)
	        shape = face_utils.shape_to_np(shape)
	        # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
	        (x, y, w, h) = face_utils.rect_to_bb(rect)
	        angle.append(np.arctan((shape[54][1]-shape[48][1])/(shape[54][0]-shape[48][0]))*180/np.pi)
	        #h1=round(max(h,w)*0.5)
	        h=round(max(h,w)*0.8)
	        h2.append(int((shape[54][0]-shape[48][0])*0.65))
	        image2 = image[shape[66][1]-h:shape[66][1]+h, shape[66][0]-h:shape[66][0]+h]
	        image2 = imutils.resize(image2, width=500)
	        #print(image2.shape)
	        out.write(image2)
	    # show the output image with the face detections + facial landmarks
	    cv2.imshow("Output", image2)
	    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()

h2= int(np.mean([i * 250 / h for i in h2]))
print('2nd Stage: Crop and Rotate Lip')
cap2=cv2.VideoCapture('output2.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out2 = cv2.VideoWriter('output3.avi',fourcc, 25.0,(112,112))
length2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

while(cap2.isOpened()):

    ret, frame = cap2.read()
    if ret==False:
        break;
    image = frame	

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
        #h2=int((shape[54][0]-shape[48][0])*0.65)
        #print(h2)
        M = cv2.getRotationMatrix2D((500/2,500/2),np.mean(angle),1)  
        image = cv2.warpAffine(image,M,(500,500))    # image2 = image[int(shape[66][1]-h2):int(shape[66][1]+h2), int(shape[66][0]-h2): int(shape[66][0]+h2)]
        image2 = image[250-h2:250+h2, 250-h2:250+h2]
        image2 = imutils.resize(image2, width=112)
        out2.write(image2)
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image2)
    cv2.waitKey(1)
cap2.release()
out2.release()
cv2.destroyAllWindows()

print('3rd Stage: Create file by input format')

file = 'output3.avi'
f_test = open('test_batch.bin', 'wb')
test_file_queue = []
vidcap = cv2.VideoCapture(file)
   
# drop the first 2 frames
vidcap.read()
vidcap.read()

count = 0  
while True:
  success,image = vidcap.read()
  if not success:
    break       
  r = image[:,:,2]
  g = image[:,:,1]
  b = image[:,:,0]
        
  r = np.reshape(r, -1)
  g = np.reshape(g, -1)
  b = np.reshape(b, -1)
        
  r = r.astype(np.int8)
  g = g.astype(np.int8)
  b = b.astype(np.int8)
      
  f_test.write(r)
  f_test.write(g)
  f_test.write(b)

  count += 1
  if count == 25:
    break          
  f_test.flush()
f_test.close()