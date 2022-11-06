from shutil import move
import socket
from imutils.video import VideoStream
import imagezmq
import time
from motors.controls import *

sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')
rpi_name = socket.gethostname()
picam = VideoStream().start()
t = time.time()

current_movement = 0
movements = {
    "w": forward,
    "a": turn_left,
    "d": turn_right,
    "s": backwards
}

while True:
    image = picam.read()
    res = sender.send_image(rpi_name, image).decode()
    if res not in "wasd" or res == "":
        gpio.cleanup()
        continue
    move = movements[res]
    move()
    if res in "ad":
        time.sleep(1/4)
        gpio.cleanup()

