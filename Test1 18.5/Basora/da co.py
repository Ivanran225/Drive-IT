import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

img = cv2.imread('Photos/calle.png')  
area_of_interest = [(400,340), [700,330], (1030, 720), (70,730)]

area_of_projection = [(70,650), [800,650], (1030, 730), (70,730)]

def project_transform(image, src, dst):
    
    tform = transform.estimate_transform('projective', np.array(src), np.array(dst))
    transformed = transform.warp(image, tform.inverse)
    resized = cv2.resize(transformed, (960, 700))

    cv2.imshow('Transformed', resized)

project_transform(img, area_of_interest, area_of_projection)
cv2.waitKey(0)