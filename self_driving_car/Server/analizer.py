import cv2
import numpy as np

def get_alignment_and_angle(thresholded_image):
    whites = []
    parallels = np.array(range(0, len(thresholded_image), 15))
    for i in parallels:
        row_whites = []
        for j in range(len(thresholded_image[0])):
            if thresholded_image[i][j] == 255:
                row_whites.append(j)
        whites.append(row_whites)

    ###############################################################################

    row_intersects = []
    for row_whites in whites:
        if len(row_whites)==0:
            row_intersects.append((0,0))
            continue
        min_ = min(row_whites)
        max_ = max(row_whites)
        center = (max_ - min_) // 2 + min_
        width = max_ - min_
        row_intersects.append((center, width))
    row_intersects = np.array(row_intersects)

    ###############################################################################

    centers = row_intersects[:,0]
    widths = row_intersects[:,1]

    non_zero = widths!=0

    parallels = parallels[non_zero]
    centers = centers[non_zero]
    widths = widths[non_zero]

    ###############################################################################

    polyfit_widths = np.polyfit(parallels, widths, 1)
    fitted_widths = polyfit_widths[0]*parallels + polyfit_widths[1]
    ratio = ratio = widths/fitted_widths

    correction_condition = [(0.7 < ratio) & (ratio < 1.3)]

    ###############################################################################

    corrected_centers = centers[correction_condition]
    corrected_parallels = parallels[correction_condition]

    polyfit_centers = np.polyfit(corrected_parallels, corrected_centers, 1)
    m, b = polyfit_centers

    ###############################################################################

    bottom_intercept = m*len(thresholded_image) + b
    top_intercept = b

    tangent = len(thresholded_image) / (top_intercept - bottom_intercept )
    angle = np.arctan(tangent) * 180 / np.pi
    car_alignment_pct = bottom_intercept/len(thresholded_image[0])

    for i, parallel in enumerate(corrected_parallels):
        cv2.circle(thresholded_image, (corrected_centers[i], parallel), 1, 0, 2)

    cv2.line(
        thresholded_image,
        (int(top_intercept),0),
        (int(bottom_intercept), len(thresholded_image)),
        0,
        2,
    )

    return angle, car_alignment_pct