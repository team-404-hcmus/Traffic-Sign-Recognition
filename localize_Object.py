import cv2
import numpy as np
import time
THREHOLD=1500
def DetectByContour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x1,y1,x2,y2=[],[],[],[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > THREHOLD:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            x1.append(x)
            y1.append(y)
            x2.append(x+w)
            y2.append(y+h)
            #cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return x1,y1,x2,y2



