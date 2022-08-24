from mss import mss
import cv2
from PIL import Image
import numpy as np


mon = {'top': 30, 'left':0, 'width':720, 'height':480}
sct = mss()
def scren_record():
    sct_img = sct.grab(mon)
    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    global img_bgr
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #cv2.imshow('test', np.array(img_bgr))

    
def scale_img():
    global image, frame
    image = np.array(img_bgr)
    frame = np.array(img_bgr)
    lineas = np.array(img_bgr)

    scale_percent = 60 # percent of original size
    width = int(720 * scale_percent / 100)
    height = int(480 * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(lineas, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Resized image", resized)

while 1:
    scren_record()

    scale_img()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
