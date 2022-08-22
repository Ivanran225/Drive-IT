# import necessary libraries

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Turn on Laptop's webcam
img = cv2.imread('Photos/god.jpg')

while True:
	
	pts1 = np.float32([[530,390], [680,380], [60,900], [1140, 900]])
	pts2 = np.float32([[400, 720], [800, 720], [400, 909], [800, 909]])
	
	# Apply Perspective Transform Algorithm
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	result = cv2.warpPerspective(img, matrix, (1399, 909))
	# Wrap the transformed image
    #resized = cv2.resize(result, (900,500))
	cv2.imshow('Original image', img) # Initial Capture
	cv2.imshow('Changed perspective', result) # Transformed Capture
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break