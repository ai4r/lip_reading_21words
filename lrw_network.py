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
import input

ckpt = tf.train.get_checkpoint_state('./weight/new_21words2/')
'''
def evaluate():
	with tf.Graph().as_default() as g:
		# image and graph calling
		image = input.inputs()
		logits = cnn.inference(image)
		top_k_predict_op = tf.argmax(logits,1)
		probabilities_op = tf.nn.softmax(logits)

		# save 
		saver = tf.train.Saver()
		sess=tf.Session()

		# Restores from checkpoint		
		saver.restore(sess, ckpt.model_checkpoint_path)	
		coord = tf.train.Coordinator()
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
'''
def evaluate():
	# image and graph calling
	image = input.inputs()
	logits = cnn.inference(image)
	top_k_predict_op = tf.argmax(logits,1)
	probabilities_op = tf.nn.softmax(logits)
	sess=tf.Session()
	coord = tf.train.Coordinator()
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
