#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import align.detect_face


with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
    
cap = cv2.VideoCapture("./media/test.mp4")
out = None

while True:
  
  ret, image = cap.read()
  
  if ret == 0:
    break

  if out is None:
    [h, w] = image.shape[:2]
    out = cv2.VideoWriter("./media/test_out_mtcnn.avi", 0, 25.0, (w, h))

  if image.ndim == 2: 
    image = facenet.to_rgb(image)
  
  image = image[:,:,0:3]

  start_time = time.time()

  # MTCNN here
  bounding_boxes, box_cord = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

  elapsed_time = time.time() - start_time

  print('inference time cost: {}'.format(elapsed_time))
  nrof_faces = bounding_boxes.shape[0]
  print("%d nrof" % (nrof_faces))
  print(bounding_boxes)
#   for rectangle in range(0,nrof_faces):
#       cv2.rectangle(image,box_cord[rectangle],(0,255,0),5)
  out.write(image)


cap.release()
out.release()
