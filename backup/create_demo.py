import cv2
import numpy as np

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