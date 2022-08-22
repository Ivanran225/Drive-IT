from mss import mss
import cv2
from PIL import Image
import numpy as np
from time import time

mon = {'top': 0, 'left':0, 'width':720, 'height':480}

sct = mss()



while 1:

    sct_img = sct.grab(mon)
    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow('test', np.array(img_bgr))

    image = np.array(img_bgr)
    #cv2.imshow('image', image)
    b,g,r = cv2.split(image)


    # difference between red and green
    # this will likely discard white
    dif = cv2.subtract(g, r)
    # cv2.imshow("diff", dif)

    ret, thresh = cv2.threshold(dif, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresh", thresh)
     
    sct_img = sct.grab(mon)
    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow('test', np.array(img_bgr))

    pts1 = np.float32([[240,140], [480,140], [1,480], [720, 480]])
    pts2 = np.float32([[1, 1], [480, 1], [1, 480], [720, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(np.array(img_bgr), matrix, (720, 480))
    # Wrap the transformed image
    #cv2.imshow('Original image', img) # Initial Capture
    cv2.imshow('Changed perspective', result) # Transformed Capture
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break           



