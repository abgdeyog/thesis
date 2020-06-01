import cv2 as cv
import numpy as np
# method for optical flow extraction between two consequtive frames
def extract_the_optical_flow(frame_1, frame_2):
    # convert the images to gray scale
    prvs = cv.cvtColor(frame_1, cv.COLOR_BGR2GRAY)
    next = cv.cvtColor(frame_2, cv.COLOR_BGR2GRAY)

    # extracting the dense optical flow
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # splitting to the x and y axis optical flow images
    gray_x = np.array(flow[..., 0])
    gray_x_to_write = cv.convertScaleAbs(gray_x, alpha=255.0)
    gray_y = np.array(flow[..., 1])
    gray_y_to_write = cv.convertScaleAbs(gray_y, alpha=255.0)
    return gray_x_to_write, gray_y_to_write
