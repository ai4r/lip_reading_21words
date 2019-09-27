import cv2
import time
import imutils
from imutils import face_utils
import numpy as np
import dlib
import os
import lrw_network as network
import cnn2 as cnn
import input
import vad_demo as vad
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# input files

input_avi = 'input_avi/demo_new_pretrained.mp4'
cap = cv2.VideoCapture(input_avi)  # for stored file
#cap = cv2.VideoCapture(0) # for webcam


word_class_en= ('get1', 'hi1', 'hi2', 'hi3', 'no1', 'no2', 'no3', 
            'wait1', 'wait2', 'wait3', 'wait4', 'what1', 'what2', 'what3',
            'where1', 'yes1', 'yes2', 'yes3', 'yes4', 'yes5', 'yes6')
word_class = ('가져와', '안녕','안녕하세요','반갑습니다','아니','아니야','싫어',
             '기다려','잠깐','잠깐만','그만','뭐라고','다시말해봐','다시말해줘',
             '어디있지','맞아','그래','그렇지','네','예','응')

test_batch_name= 'test_batch.bin'
shape_predictor = 'weight/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

fps = round(cap.get(cv2.CAP_PROP_FPS))
ret, frame= cap.read()
frame= imutils.resize(frame, height=960)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ckpt = tf.train.get_checkpoint_state('./weight/new_21words2/')
## vad case
out = cv2.VideoWriter('demo_result.avi',fourcc, 10,(frame.shape[1],frame.shape[0]))
start_i, end_i, mid_i = vad.generate(input_avi, fps) # VAD of file

## vvad case


def crop_lip(frame, mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects): 
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        h_len = int(0.6*(shape[54][0]-shape[48][0]))
        center = [int(0.5*(shape[62][1]+shape[66][1])), int(0.5*(shape[62][0]+shape[66][0]))]
        if mode =='second':
            angle.append(np.arctan((shape[54][1]-shape[48][1])/(shape[54][0]-shape[48][0]))*180/np.pi)
            return h_len, center, angle
        else:
            return h_len, center, rects


mode='first'
count=49
N_frame =25
cnt=0
j=0
frame_cnt=0
demo_font = cv2.FONT_HERSHEY_SIMPLEX


####LHW_INITIALIZE_VARIABLE
image = None
logits = None
top_k_predict_op = None
probabilities_op = None

saver = None
sess = None
coord = None

####LHW INITIALIZE_FUNCTION

def init():
    start_time = time.time()
    print('Init start...')
    global image, logits, top_k_predict_op, probabilities_op, saver, sess, coord
    image = input.inputs()
    logits = cnn.inference(image)
    top_k_predict_op = tf.argmax(logits, 1)
    probabilities_op = tf.nn.softmax(logits)

    saver = tf.train.Saver()
    sess = tf.Session()

    # Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    print('Init finish...')
    print('Initialize time: ' + str(time.time() - start_time))

def eval():
    # image and graph calling
    image = input.inputs()
    logits = cnn.inference(image)
    top_k_predict_op = tf.argmax(logits,1)
    probabilities_op = tf.nn.softmax(logits)
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        # Session running
        image, test_labels, test_prob = sess.run([image, top_k_predict_op, probabilities_op])
        return image, test_labels, test_prob
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
    coord.request_stop()

init()


print('start')
while(cap.isOpened()):
    # start frame
    ret, frame = cap.read()
    if ret==False:
        break;
    frame = imutils.resize(frame, height=480)
    frame_cnt +=1

    #if cv2.waitKey(50) & 0xFF == ord('1'):
    if frame_cnt in start_i:
        mode='second'
        angle =[]
        h2 =[]
        frame_seq=[]
        cnt=0

    if mode == 'first' :
        h_len, center, rects = crop_lip(frame, mode)

        if len(rects)>0:
            cv2.rectangle(frame, (center[1]-h_len, center[0]-h_len), (center[1]+h_len, center[0]+h_len), (0, 0, 255), 2)
        cv2.putText(frame,'To quit this program, press ESC, To start saving, press 1', (15,20), demo_font, 0.7, (0, 255, 0), 2)

    if mode == 'third':
        cnt+=1
        h_len, center, rects = crop_lip(frame, mode)
        if len(rects)>0:
            cv2.rectangle(frame, (center[1]-h_len, center[0]-h_len), (center[1]+h_len, center[0]+h_len), (0, 0, 255), 2)
        cv2.putText(frame,"Predicted Label: {}({:2.1%})".format(word_class_en[test_labels[0]][:-1],test_prob[0][test_labels[0]]), (15,20), demo_font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame,'Time: %2.2f sec' % (final_time), (15,50), demo_font, 0.7, (0, 255, 0), 2)
        
        if cnt ==20 or cv2.waitKey(50) & 0xFF == ord('2'):
            mode = 'first'
            cnt=0
            j+=1

    if mode == 'second':    
        cnt+=1
        h_len, center, angle = crop_lip(frame, mode)

        lip_frame = frame[center[0]-h_len:center[0]+h_len,center[1]-h_len:center[1]+h_len]
        lip_frame = imutils.resize(lip_frame, width=112)
        frame_seq.append(lip_frame)

        cv2.rectangle(frame, (center[1]-h_len, center[0]-h_len), (center[1]+h_len, center[0]+h_len), (0, 0, 255), 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-2, frame.shape[0]-2), (0, 0, 255), 2)
        cv2.putText(frame,'frame : {}'.format(cnt), (15,20), demo_font, 0.7, (0, 255, 0), 2)      

        #if cnt==49 or (cnt>25 and cv2.waitKey(50) & 0xFF == ord('2')):
        if cnt==max(25, end_i[j]-start_i[j]):

            #print('Preprocess is Loading')
            start_time = time.time()
            count=cnt 

            # Frame Count method 
            if count>=N_frame:
                if count % 2 ==0:
                    frperiod =int(count/N_frame)
                    frame_N= range(int(count/2-(((N_frame-1)/2)*frperiod)),int(count/2+(((N_frame-1)/2)*frperiod)+1),frperiod)
                else:
                    frperiod =int((count+1)/N_frame)
                    frame_N= range(int((count+1)/2-(((N_frame-1)/2)*frperiod)),int((count+1)/2+(((N_frame-1)/2)*frperiod)+1),frperiod)

            # make input structure by 'bin'            
            #out2 = cv2.VideoWriter('output2.avi',fourcc, 25.0,(112,112))
            f = open(test_batch_name, 'wb')
            for i in frame_N:
                frame2 = frame_seq[i-1]
                M = cv2.getRotationMatrix2D((112/2,112/2),angle[i-1],1)  
                frame2 = cv2.warpAffine(frame2,M,(112,112))
                #out2.write(frame2)

                r = np.reshape(frame2[:,:,2], -1)
                g = np.reshape(frame2[:,:,1], -1)
                b = np.reshape(frame2[:,:,0], -1)
                r = r.astype(np.int8)
                g = g.astype(np.int8)
                b = b.astype(np.int8)            
                f.write(r)
                f.write(g)
                f.write(b)
                f.flush()
            f.close()
            #out2.release()

            image, test_labels, test_prob=eval()
            
            
            final_time= time.time()-start_time

            # Result Making
            print("Predicted Label: {}({:2.1%})".format(word_class[test_labels[0]],test_prob[0][test_labels[0]]))
            #print("Predicted Label: {}({:2.1%})".format(word_class[test_label[0]],test_pro[0][test_label[0]]))
            for i in range(len(word_class)):
                if test_prob[0][i]>0.05:
                #if test_pro[0][i]>0.05:
                    print('Label {}: {:2.1%}'.format(word_class_en[i].ljust(9),test_prob[0][i]))     
                    #print('Label {}: {:2.1%}'.format(word_class_en[i].ljust(9),test_pro[0][i]))     
            print('Prediction finish with %2.2f seconds' % (final_time))
            

            mode='third'
            cnt=0
    frame = imutils.resize(frame, height=960)
    out.write(frame)


    if cv2.waitKey(50) & 0xFF == 27:
        print('Demo Finish')
        ####LHW SESSION CLOSE
        sess.close()
        break;
####LHW SESSION CLOSE WHEN SESS is ALIVE
if(not sess is None):
    sess.close()

cap.release()
cv2.destroyAllWindows()
out.release()
