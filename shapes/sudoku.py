import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def contour(img):

    img = cv.medianBlur(img,5)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)

    return th3

def detect_shape(th, original):
    _,contours,h = cv.findContours(th,1,2)

    for cnt in contours:
        approx = cv.approxPolyDP(cnt,0.02*cv.arcLength(cnt,True),True)
        print(len(approx))
        if len(approx) == 10:
            print("square")
            cv.drawContours(original,[cnt],0,(0,0,255),-1)
    return original

def detect_lines(img):

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
			min_line_length, max_line_gap)

    #line_image = img

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    return line_image

#img = cv.imread('test.jpg',0)
img = cv.imread('test.jpg')
#cont = contour(img)
#img_with_shapes = detect_shape(cont, img)
lines = detect_lines(img)
cv.imshow('',lines)
cv.waitKey(0)
