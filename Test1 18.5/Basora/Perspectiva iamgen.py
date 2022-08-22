# import necessary libraries

import cv2
import numpy as np

# Turn on Laptop's webcam
img = cv2.imread('Photos/god.jpg')

while True:
	
	pts1 = np.float32([(400,340), (700,330), (1030, 720), (70,730)])
	pts2 = np.float32([(70,300),(800,300), (1030, 730), (70,730)])
	
	# Apply Perspective Transform Algorithm
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	result = cv2.warpPerspective(img, matrix, (1200, 700))
	print(matrix)
	# Wrap the transformed image
	cv2.imshow('Original image', img) # Initial Capture
	cv2.imshow('Changed perspective', result) # Transformed Capture
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break



""""
import cv2 

cap = cv2.VideoCapture(0)

while cap.isOpened(): 
    _, frame = cap.read()

    cv2.imshow(('Frame', frame))
+
"""