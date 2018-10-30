import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf

from picamera import PiCamera
from io import BytesIO,StringIO
import time
from collections import defaultdict
from PIL import Image
import boto3
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from time import gmtime, strftime


import pyrebase

config = {
  "apiKey": "AIzaSyAYGVQ1EdhziACQuIZos639z1sn1yk1Xbk",
  "authDomain": "dooralert-4eee3.firebaseapp.com",
  'projectId': "dooralert-4eee3",
  "databaseURL": "https://dooralert-4eee3.firebaseio.com",
  "storageBucket": "dooralert-4eee3.appspot.com"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

database = firebase.database()

stream = BytesIO()
camera = PiCamera()


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile('ssd_mobilenet_v1_coco_2018_01_28' + '/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      keys = ['num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes']
      for key in keys:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      output = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})


  return output['detection_classes'][0].astype(np.uint8),output['detection_scores'][0]


def sendEmail(frame,cur_time,notMaster=False):
    sender_mail = 'dooralert.iot@gmail.com'
    msg = MIMEMultipart()
    msg['From'] = sender_mail
    msg['To'] = sender_mail
    msg['Subject'] = "Door Alert" + ' ' + cur_time

    body = ''

    if notMaster:
        body = 'Someone appears at your door'
        img = MIMEImage(frame)
        msg.attach(img)
    else:
        body = 'House Owner (for demo only)'

    msg.attach(MIMEText(body, 'plain'))
    text = msg.as_string()
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_mail, "Shadow1994")
    server.sendmail(sender_mail, sender_mail, text)
    server.quit()

def updateToFirebase(cur_time,cur_time_sec):

    storage.child("images/" + cur_time + ".jpg").put("target.jpg")

    url = storage.child("images/" + cur_time + ".jpg").get_url(None)

    data = {'time' : cur_time[17:-4], 'imgUrl': url, 'date': cur_time[:17], 'time_in_sec' : cur_time_sec}

    database.child("events").push(data)


def sendOperation(cur_time,cur_time_sec,notMaster=False):

    with open('target.jpg','rb') as frame:
        if notMaster:
            sendEmail(frame.read(),cur_time,notMaster)
            updateToFirebase(cur_time,cur_time_sec)
        else:
            sendEmail(frame.read(),cur_time)

    time.sleep(1)

def contains_faces(target_64,rekognition):
    response = rekognition.detect_faces(
        Image={
                'Bytes': target_64
            }
        )
    return len(response['FaceDetails']) > 0


def runDectection():

    BUCKET = "amazon-rekognition"

    master_64 = ""


    with open('master.jpg', "rb") as image_file:
        master_64 = base64.decodestring(base64.b64encode(image_file.read()))

        while(True):

            try:


                print ("new frame")

                time.sleep(8)

                print ('capture')

                camera.capture(stream, format='jpeg')
                stream.seek(0)
                frame = Image.open(stream)


                frame.save('target.jpg')
                stream.seek(0)
                stream.truncate()

                detection_classes,detection_scores = run_inference_for_single_image(frame, detection_graph)
                for index,value in enumerate(detection_classes):
                    if detection_scores[index] > 0.7 and value == 1:

                        cur_time = time.strftime("%a, %d %b %Y %I:%M:%S %p %Z", time.localtime())
                        cur_time_sec = time.time()

                        with open('target.jpg', "rb") as image_file:
                            body = 'Someone appears at your door'
                            target_64 = base64.decodestring(base64.b64encode(image_file.read()))

                            rekognition = boto3.client("rekognition", 'us-east-1')


                            if contains_faces(target_64,rekognition):

                                response = rekognition.compare_faces(SourceImage={"Bytes": master_64},
                                              TargetImage={"Bytes": target_64},
                    	                     SimilarityThreshold=80
                    	                     )
                                for match in response['FaceMatches']:

                                    if match['Similarity'] > 0.9:
                                        print ('detect')
                                        sendOperation(cur_time,cur_time_sec)
                                        break
                                    else:

                                        print ('match')
                                sendOperation(cur_time,cur_time_sec,True)
                            else:
                                sendOperation(cur_time,cur_time_sec,True)
            except KeyboardInterrupt:
                camera.close()


runDectection()
