from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import cv2
import os

import numpy as np
import tensorflow as tf
import cnn

eval_interval_secs = 1
NUM_CLASSES = 10
word_class = ('응','맞아','그렇지','예','잠깐만','멈춰','아니','싫어','안녕','반가워')
word_class_en = ('Ueng','Maja','Greochi','Ye','JamGanMan','Meomcheo','Ani','Sireo','Annyeong','Bangawo')

def eval_once(saver, top_k_predict_op, probabilities_op, image): 
	with tf.Session() as sess:
		# Restores from checkpoint
		ckpt = tf.train.get_checkpoint_state('./weight')
		saver.restore(sess, ckpt.model_checkpoint_path)
		global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

		# Start the queue runners.
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
			# Session Running
			image, test_labels, test_prob = sess.run([image, top_k_predict_op, probabilities_op])
			return image, test_labels, test_prob


		except Exception as e:  # pylint: disable=broad-except
			coord.request_stop(e)
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=1)

def evaluate():
	with tf.Graph().as_default() as g:
		# Get image
		image = cnn.inputs()

		# Build a Graph that computes the logits predictions from the inference model.
		logits = cnn.inference(image, Train_Flag = False)
		#top_k_op = tf.nn.in_top_k(logits, labels, 1)
		top_k_predict_op = tf.argmax(logits,1)
		probabilities_op = tf.nn.softmax(logits)

		# Restore the moving average version of the learned variables for eval.
		variable_averages = tf.train.ExponentialMovingAverage(cnn.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		image, test_labels, test_prob= eval_once(saver, top_k_predict_op, probabilities_op, image)
		return image, test_labels, test_prob