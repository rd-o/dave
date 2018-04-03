import numpy as np
import cv2

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)

cv2.imshow('',thresh)
cv2.waitKey(0)
_,contours,h = cv2.findContours(thresh,1,2)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    print(len(approx))
    if len(approx)==4:
        print("square")
        cv2.drawContours(img,[cnt],0,(0,0,255),-1)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
