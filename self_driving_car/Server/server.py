import cv2
import imagezmq
import keyboard
from analizer import get_alignment_and_angle
import time

image_hub = imagezmq.ImageHub()
autopilot = False

while True:  # show streamed images until Ctrl-C
    rpi_name, frame = image_hub.recv_image()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (25, 25), 0)
    ret, thresh = cv2.threshold(blurred_gray, 205, 255, cv2.THRESH_BINARY)

    angle = None
    alignment = None
    
    try:
        angle, alignment = get_alignment_and_angle(thresh)
    except:
        pass

    print("ANGLE:", angle, "ALIGNMENT:", alignment)
    cv2.imshow(f"{rpi_name}", frame)
    cv2.imshow(f"{rpi_name}_thresholded", thresh)

    cv2.waitKey(1)
    
    alignment_limits = (0.4, 0.6)
    angle_limits = (-80, 80)

    movement = ""

    if alignment is None:
        alignment = 0.5

    if angle is None:
        angle = 90

    if alignment < alignment_limits[0]:
        movement = "a"
    elif alignment > alignment_limits[1]:
        movement = "d"
    else:
        movement = "w"

    if not autopilot:
        movement = ""

    for key in ["w", "a", "s", "d"]:
        if keyboard.is_pressed(key):
            movement = key

    if keyboard.is_pressed("q"):
        autopilot = not autopilot

    image_hub.send_reply(movement.encode("utf-8"))