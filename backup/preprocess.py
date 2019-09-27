import cv2
import time
import imutils
from imutils import face_utils
import numpy as np
import dlib
import time
import os
import tensorflow as tf

# input files and parameters
shape_predictor = 'weight/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
sess =tf.Session()
# open and lip distance conditions

def liplandmark_process(file_name, cap, out1, out2, out3):

    cnt=0
    N_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    angle = []
    frame_seq =[]
    mode=0
    while(cap.isOpened()):
        
        # read frame
        ret, frame = cap.read()
        cnt+=1
        if ret==False:
            break;
        #frame = imutils.resize(frame, height=720)


        # Detect facial landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for (i, rect) in enumerate(rects): 
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            h_len1 = round(max(h,w)*0.8)
            h_len2 = int(0.6*(shape[54][0]-shape[48][0]))
            center = [int(0.5*(shape[62][1]+shape[66][1])), int(0.5*(shape[62][0]+shape[66][0]))]
            if center[0]+h_len1>frame.shape[0]:
                h_len1=frame.shape[0]-center[0]
            angle.append(np.arctan((shape[54][1]-shape[48][1])/(shape[54][0]-shape[48][0]))*180/np.pi)
            if h_len1 <=0:
                mode =1
                break
            else:
                frame1 = frame[center[0]-h_len1:center[0]+h_len1, center[1]-h_len1:center[1]+h_len1]
                frame1 = imutils.resize(frame1, width=500)
                frame2 = frame[center[0]-h_len2:center[0]+h_len2, center[1]-h_len2:center[1]+h_len2]
                frame2 = imutils.resize(frame2, width=120)
                frame_seq.append(frame2)
                out1.write(frame1)
                out2.write(frame2)     

    if mode ==0:
        if len(frame_seq) == N_frame:
            for i in range(len(frame_seq)):
                #print(i)
                frame = frame_seq[i]
                M = cv2.getRotationMatrix2D((120/2,120/2),np.mean(angle),1)
                frame3 = cv2.warpAffine(frame,M,(120,120))
                out3.write(frame3)
            return N_frame, mode            
        else:
            print(len(frame_seq), N_frame)
            return N_frame, mode  

    else:
        return 0, mode


# main code
def generate():     
    dir_name = '../dataset_lipread/test'
    save_dir_name1 = '../dataset_lipread/filter1_old2'
    save_dir_name2 = '../dataset_lipread/filter2_old2'
    save_dir_name3 = '../dataset_lipread/filter3_old2'

    file_name_list=os.listdir(dir_name)
    #file_name_list=[file for file in file_name_list if '013' in file]  
    #file_name_list=['002_old_d_get1_1.mp4']    

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    with tf.Session() as sess:
        for file_name in file_name_list[10969:]:
            
            # inputs : file / Path arrangement
            ab=time.time()
            full_path = os.path.join(dir_name, file_name)
            save_full_path1= os.path.join(save_dir_name1, file_name)
            save_full_path2= os.path.join(save_dir_name2, file_name)
            save_full_path3= os.path.join(save_dir_name3, file_name)

            # Preprocessing
            cap = cv2.VideoCapture(full_path) 
            out1 = cv2.VideoWriter('%s.avi' % (save_full_path1[:-4]), fourcc, 25.0, (500,500))
            out2 = cv2.VideoWriter('%s.avi' % (save_full_path2[:-4]), fourcc, 25.0, (120,120))
            out3 = cv2.VideoWriter('%s.avi' % (save_full_path3[:-4]), fourcc, 25.0, (120,120))
            
            # main process
            N_frame, mode=liplandmark_process(file_name, cap, out1, out2, out3)

            # result arrangement 
            cap.release()
            out1.release()
            out2.release()
            out3.release()
            if mode ==0:
                print("Name: %s, # Frame: %d (%2.2fs)" % (file_name, N_frame, time.time()-ab))
            else:
                print('%s: failure' %file_name)
    
def main():
    result=generate()
    
if __name__ == '__main__':    
    main()