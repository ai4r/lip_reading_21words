import cv2
import time
import imutils
from imutils import face_utils
import numpy as np
import dlib
import os
import lrw_network as network

# input files
test_batch_name= 'test_batch.bin'
input_avi = 'lhw_1.mp4'
cap = cv2.VideoCapture(input_avi)  # for stored file
#cap = cv2.VideoCapture(0) # for webcam
os.environ["CUDA_VISIBLE_DEVICES"]="1"
word_class = ('응','맞아','그렇지','예','잠깐만','멈춰','아니','싫어','안녕','반가워')
word_class_en = ('Ueng','Maja','Greochi','Ye','JamGanMan','Meomcheo','Ani','Sireo','Annyeong','Bangawo')
word_class_en2 = ('Yes','right','right','Yes','wait','Stop','No','No','Hi','Nice to meet you')

shape_predictor = 'weight/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

print('Select to speak: 응 / 맞아 / 그렇지 / 예 / 잠깐만 / 멈춰 / 아니 / 싫어 / 안녕 / 반가워')
print('if you want to start saving, press 1')

mode='first'
count=49
N_frame =25
cnt=0

# To find width/height size
ret, frame= cap.read()
frame= imutils.resize(frame, height=480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('demo_result.avi',fourcc, 10.0,(frame.shape[1],frame.shape[0]))

while(cap.isOpened()):
    # start frame
    ret, frame = cap.read()
    if ret==False:
        break;
    frame = imutils.resize(frame, height=480)

    if cv2.waitKey(50) & 0xFF == ord('1'):
        mode='second'
        angle =[]
        h2 =[]
        frame_seq=[]
        cnt=0

    if mode == 'first' :
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for (i, rect) in enumerate(rects): 
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            h_len = int(0.6*(shape[54][0]-shape[48][0]))
            center = [int(0.5*(shape[62][1]+shape[66][1])), int(0.5*(shape[62][0]+shape[66][0]))]
        if len(rects)>0:
            cv2.rectangle(frame, (center[1]-h_len, center[0]-h_len), (center[1]+h_len, center[0]+h_len), (0, 0, 255), 2)
        cv2.putText(frame,'To quit this program, press ESC, To start saving, press 1', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if mode == 'third':
        cnt+=1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for (i, rect) in enumerate(rects): 
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            h_len = int(0.6*(shape[54][0]-shape[48][0]))
            center = [int(0.5*(shape[62][1]+shape[66][1])), int(0.5*(shape[62][0]+shape[66][0]))]

        if len(rects)>0:
            cv2.rectangle(frame, (center[1]-h_len, center[0]-h_len), (center[1]+h_len, center[0]+h_len), (0, 0, 255), 2)
        cv2.putText(frame,"Predicted Label: {}({:2.1%})".format(word_class_en2[test_labels[0]],test_prob[0][test_labels[0]]), (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame,'Time: %2.2f sec' % (final_time), (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if cnt ==25 or cv2.waitKey(50) & 0xFF == ord('2'):
            mode = 'first'
            cnt=0

    if mode =='second':    
        cnt+=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        # Crop lip position and make angle list   
        for (i, rect) in enumerate(rects): 
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            h_len = int(0.6*(shape[54][0]-shape[48][0]))
            center = [int(0.5*(shape[62][1]+shape[66][1])), int(0.5*(shape[62][0]+shape[66][0]))]
            angle.append(np.arctan((shape[54][1]-shape[48][1])/(shape[54][0]-shape[48][0]))*180/np.pi)

        lip_frame = frame[center[0]-h_len:center[0]+h_len,center[1]-h_len:center[1]+h_len]
        lip_frame = imutils.resize(lip_frame, width=112)
        frame_seq.append(lip_frame)

        cv2.rectangle(frame, (center[1]-h_len, center[0]-h_len), (center[1]+h_len, center[0]+h_len), (0, 0, 255), 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-2, frame.shape[0]-2), (0, 0, 255), 2)
        cv2.putText(frame,'Press 2 for early quit', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame,'frame : {}'.format(cnt), (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

        if cnt==49 or (cnt>25 and cv2.waitKey(50) & 0xFF == ord('2')):

            print('Preprocess is Loading')
            start_time = time.time()
            count=cnt 

            # Frame Count method 
            if count>N_frame:
                if count % 2 ==0:
                    frperiod =int(count/N_frame)
                    frame_N= range(int(count/2-(((N_frame-1)/2)*frperiod)),int(count/2+(((N_frame-1)/2)*frperiod)+1),frperiod)
                else:
                    frperiod =int((count+1)/N_frame)
                    frame_N= range(int((count+1)/2-(((N_frame-1)/2)*frperiod)),int((count+1)/2+(((N_frame-1)/2)*frperiod)+1),frperiod)

            #out2 = cv2.VideoWriter('output2.avi',fourcc, 25.0,(112,112))
            # make input structure by 'bin'
            f = open(test_batch_name, 'wb')
            for i in frame_N:
                frame2 = frame_seq[i-1]
                M = cv2.getRotationMatrix2D((112/2,112/2),np.mean(angle[i-1]),1)  
                frame2 = cv2.warpAffine(frame2,M,(112,112))
                #out2.write(frame2)

                r = frame2[:,:,2]
                g = frame2[:,:,1]
                b = frame2[:,:,0]
                r = np.reshape(r, -1)
                g = np.reshape(g, -1)
                b = np.reshape(b, -1)
                r = r.astype(np.int8)
                g = g.astype(np.int8)
                b = b.astype(np.int8)            
                f.write(r)
                f.write(g)
                f.write(b)
                f.flush()
            f.close()
            #out2.release()

            # Session running by network
            image, test_labels, test_prob=network.evaluate()
            final_time= time.time()-start_time

            # Result Making
            print("Predicted Label: {}({:2.1%})".format(word_class[test_labels[0]],test_prob[0][test_labels[0]]))
            for i in range(len(word_class)):
                if test_prob[0][i]>0.01:
                    print('Label {}: {:2.1%}'.format(word_class_en[i].ljust(9),test_prob[0][i]))     
            print('Prediction finish with %2.2f seconds' % (final_time))

            mode='third'
            cnt=0

    cv2.imshow('frame',frame)
    out.write(frame)

    if cv2.waitKey(50) & 0xFF == 27:
        print('Demo Finish')
        break;

cap.release()
cv2.destroyAllWindows()
out.release()