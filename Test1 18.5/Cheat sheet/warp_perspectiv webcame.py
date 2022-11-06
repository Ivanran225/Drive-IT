# import necessary libraries

import cv2
import numpy as np

# Turn on Laptop's webcam
cap = cv2.VideoCapture(0)

while True:
	
	_, frame = cap.read()

	pts1 = np.float32([[210,140], [350,140], [180,370], [380, 370]])
	pts2 = np.float32([[200, 100], [400, 100], [200, 300], [400, 300]])
	
	# Apply Perspective Transform Algorithm
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	result = cv2.warpPerspective(frame, matrix, (600, 500))
	# Wrap the transformed image
    #resized = cv2.resize(result, (900,500))
	cv2.imshow('Original image', frame) # Initial Capture
	cv2.imshow('Changed perspective', result) # Transformed Capture
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break