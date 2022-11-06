# run this program on each RPi to send a labelled image stream
import socket
import time
import cv2
from imutils.video import VideoStream
import imagezmq
cap = cv2.VideoCapture(1)

sender = imagezmq.ImageSender(connect_to='tcp://locahost:5555')
rpi_name = socket.gethostname() # send RPi hostname with each image
cap = VideoStream(usePiCamera=True).start()
time.sleep(2.0)  # allow camera sensor to warm up
while True:  # send images as stream until Ctrl-C
   image = cap
   sender.send_image(rpi_name, image)