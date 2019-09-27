import cv2
import time
import imutils
from imutils import face_utils
import numpy as np
import dlib
import time
import matplotlib.pyplot as plt
import os

# input files and parameters
shape_predictor = 'weight/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# open and lip distance conditions
'''
dist1 : lip difference frame by frame 
dist2 : lip distance for each frame
dist3 : lip distance difference frame by frame
'''
def liplandmark_process(cap):
    dist1=[]
    dist2=[]
    dist3=[]
    dist_ele=[]
    cnt=0
    N_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened()):
        # read frame
        ret, frame = cap.read()
        cnt+=1
        if ret==False:
            break;
        frame = imutils.resize(frame, height=400)

        # Detect facial landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for (i, rect) in enumerate(rects): 
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            dist_ele.append(shape[60:67][:])
            dist2.append(abs(shape[66][1]-shape[62][1]))
    
    dist1_1=np.array(dist_ele[1:])-np.array(dist_ele[:-1])
    for i in range(dist1_1.shape[0]):
        dist1.append(np.sum([x**2 for x in dist1_1[i]]))
    dist3=np.array(dist2[1:])-np.array(dist2[:-1])
    return dist1[:-10], dist2[:-10], dist3[:-10]

# main code
def generate():     
    #dir_name = './input_avi'    
    dir_name = '../dataset_lipread/1st_trial'
    save_dir_name = './1st_trial_vvad'
    file_name_list=os.listdir(dir_name)    
    #file_name_list=['001_old_d_get1.mp4']    
    #file_name = 'kgy_1.mp4'
    for file_name in file_name_list:
        
        # inputs : file / videoCapture / figure open
        print(file_name)
        ab=time.time()
        full_path = os.path.join(dir_name, file_name)       
        
        # Preprocessing
        cap = cv2.VideoCapture(full_path) 
        dist1,dist2,dist3 = liplandmark_process(cap)
        cap.release()

        # plotting result and saving
        t1=np.arange(0,len(dist1),1)
        t2=np.arange(0,len(dist2),1)

        plt.figure(figsize=(17,10))

        plt.subplot(3,1,1)
        plt.plot(t1,dist1)
        plt.title('lip difference',fontsize='10')
        plt.ylim((0,40))

        plt.subplot(3,1,2)
        plt.plot(t2,dist2)
        plt.title('lip distance',fontsize='10')
        plt.ylim((0,20))

        plt.subplot(3,1,3)
        plt.plot(t1,dist3)
        plt.title('lip distance difference',fontsize='10')
        plt.ylim((-5,5))

        #plt.show()
        plt.savefig('{}/{}.png'.format(save_dir_name, file_name[:-4]))
        plt.close()

        print("Elapsed Time: %2.2f sec" % (time.time()-ab))
    
def main():
    result=generate()
    
if __name__ == '__main__':    
    main()