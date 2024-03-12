import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


def LiveCamEdgeDetection_Canny(image_color):

    threshold_1 = 30
    threshold_2 = 80
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(image_gray, threshold_1, threshold_2)

    return  canny


def LiveCamEdgeDetection_Laplace(image_color):
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)

    return laplacian


def LiveCamEdgeDetection_sobely(image_color):
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    y_sobel = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=7)

    return y_sobel

"""Initializing webcam"""
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Live Edge Detection',
               LiveCamEdgeDetection_Canny(frame))

    cv2.imshow('webcam Video', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()