import cv2
import os
import time
from analizer import get_alignment_and_angle
frames = os.listdir("frames")
frames.sort()

for frame in frames:
    image = cv2.imread(f"frames/{frame}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (21, 21), 0)
    ret,thresh1 = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
    try:
        angle, alignment = get_alignment_and_angle(thresh1)
        print("Angle:", angle, "Alignment:", alignment)
    except:
        pass

    cv2.imshow("image", image)
    cv2.imshow("thresholded", thresh1)

    time.sleep(0.1)
    cv2.waitKey(1)