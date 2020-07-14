import numpy as np
import dlib
import os
import cv2
import time
import imutils
from imutils import face_utils
import tensorflow as tf
import cnn2 as cnn
#ckpt = tf.train.get_checkpoint_state('./weight/new_21words2/')
#import cnn
#ckpt = tf.train.get_checkpoint_state('./weight/new_21words/')
#ckpt = tf.train.get_checkpoint_state('../lipreading/orig/train/')

# path of the weight file
ckpt = tf.train.get_checkpoint_state('weight/demo/')

#data_dir = '../lipreading/filter3_old2'
# folder of the data
data_dir = 'dataset/filter3_old2'
#data_dir = 'dataset/old_clip'


os.environ["CUDA_VISIBLE_DEVICES"]="1"
class_num = 21
word_class_en= ('get1', 'hi1', 'hi2', 'hi3', 'no1', 'no2', 'no3', 
            'wait1', 'wait2', 'wait3', 'wait4', 'what1', 'what2', 'what3',
            'where1', 'yes1', 'yes2', 'yes3', 'yes4', 'yes5', 'yes6')
word_class = ('가져와', '안녕','안녕하세요','반갑습니다','아니','아니야','싫어',
             '기다려','잠깐','잠깐만','그만','뭐라고','다시말해봐','다시말해줘',
             '어디있지','맞아','그래','그렇지','네','예','응')
class_dict= {'get1':1, 'hi1':2, 'hi2':3, 'hi3':4, 'no1':5, 'no2':6, 'no3':7, 
            'wait1':8, 'wait2':9, 'wait3':10, 'wait4':11, 'what1':12, 'what2':13, 'what3':14,
            'where1':15, 'yes1':16, 'yes2':17, 'yes3':18, 'yes4':19, 'yes5':20, 'yes6':21}
class_dict2= {'get':1, 'hi':2, 'no':3, 'wait':4, 'what':5, 'where':6, 'yes':7}
test_number = [1,3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
#test_number = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
test_file_queue = []
test_file_cnt = np.zeros(21)
cfmatrix1 = np.zeros((21,21))
cfmatrix2 = np.zeros((7,7))
true_count1=0
true_count2=0


def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img

def rep_frame(count): 
    if count % 2 ==0:
        frperiod =int(count/25)
        frame_N= range(int(count/2-(((25-1)/2)*frperiod)),int(count/2+(((25-1)/2)*frperiod)+1),frperiod)
    else:
        frperiod =int((count+1)/25)
        frame_N= range(int((count+1)/2-(((25-1)/2)*frperiod)),int((count+1)/2+(((25-1)/2)*frperiod)+1),frperiod)
    return frame_N

def preprocess_lip(frame_N, frame_seq):             
    reader=np.array([])
    for i in frame_N:
        frame = frame_seq[i-1]
        r = np.reshape(frame[:,:,2], -1)
        g = np.reshape(frame[:,:,1], -1)
        b = np.reshape(frame[:,:,0], -1)
        r = r.astype(np.int8)
        g = g.astype(np.int8)
        b = b.astype(np.int8)  
        reader=np.concatenate((reader,r), axis=None)
        reader=np.concatenate((reader,g), axis=None)
        reader=np.concatenate((reader,b), axis=None)
    reader = reader.astype(np.uint8)
    reader = np.reshape(reader, (25, 3, 112, 112))
    reader = np.transpose(reader, (0,2,3,1))
    reader = reader.astype(np.int32)
    image = []
    for i in range(25):
        image.append(np.expand_dims(standardize(reader[i, :, :, :]), 0))
    image = np.concatenate(image,0)
    image = np.expand_dims(image, 0)
    image = image.astype(np.float16)
    return image




dir_list = os.listdir(data_dir)
for file in dir_list:
    split_str = file.split('_')
    file_number = int(split_str[0])
    label= int(class_dict[split_str[3]])

    full_path_file_name = os.path.join(data_dir, file)
    if file_number in test_number:
        test_file_cnt[label-1]+=1
        test_file_queue.append(full_path_file_name)


# WEIGTH IMPORTING AND INITIALIZATION
g=tf.Graph()
with g.as_default():
    # Restore the graph and weight
    #img=tf.compat.v1.placeholder(tf.float16, shape=(1,25,112,112,3))
    img=tf.placeholder(tf.float16, shape=(1,25,112,112,3))
    logits = cnn.inference(img)
    top_k_predict_op = tf.argmax(input=logits,axis=1)
    probabilities_op = tf.nn.softmax(logits)
    #saver = tf.compat.v1.train.Saver()
    saver = tf.train.Saver()
    #with tf.compat.v1.Session(graph=g) as sess:
    with tf.Session(graph=g) as sess:
        saver.restore(sess, ckpt.model_checkpoint_path) 
        coord = tf.train.Coordinator()
        threads = []
        #for qr in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.QUEUE_RUNNERS):
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        test_labels, test_prob = sess.run([top_k_predict_op, probabilities_op], feed_dict={img: np.zeros([1,25,112,112,3])})


        
        # Run the video file
        print('START TEST')
        start_time = time.time()

        for idx, full_path_file_name in enumerate(test_file_queue):
            cap = cv2.VideoCapture(full_path_file_name)  # for stored file
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frame check
            frame_N = rep_frame(count)
            frame_seq=[]

            file_name = full_path_file_name.split('/')[-1]
            true_label = file_name.split('_')[3]

            while True:
                success, image = cap.read()
                if not success:
                    break    
                cropped_img = image[4:116, 4:116]
                frame_seq.append(cropped_img)   
            
            image = preprocess_lip(frame_N, frame_seq)
            label, prob = sess.run([top_k_predict_op, probabilities_op], feed_dict={img: image})
            predict_label = word_class_en[label[0]]
            
            t1 = int(class_dict[true_label])
            t2 = int(class_dict2[true_label[:-1]])            
            p1 = int(class_dict[predict_label])
            p2 = int(class_dict2[predict_label[:-1]])

            cfmatrix1[t1-1][p1-1]+=1
            cfmatrix2[t2-1][p2-1]+=1


            if t1==p1:
                true_count1+=1
                true_count2+=1
            elif t1 != p1 and t2 == p2:
                true_count2+=1
                #print("filename: {}, Predicted Label: {}({:2.1%})".format(file_name, predict_label, prob[0][label[0]]))
            else:
                print("Fail! filename: {}, Predicted Label: {}({:2.1%})".format(file_name, predict_label, prob[0][label[0]]))


            if idx % 50 ==0:
                print('{:2.1%} complete'.format(idx/len(test_file_queue)))
        print('FINISH TEST')
     
        final_time= round(time.time()-start_time, 2)
        print('Time: {}sec totally, {}sec per word'.format(final_time, round(final_time/len(test_file_queue),2)))
        print('accuracy of 21 words: {:2.1%}'.format(true_count1/len(test_file_queue)))
        print('accuracy of 7 case: {:2.1%}'.format(true_count2/len(test_file_queue)))
        if(not sess is None):
            sess.close()