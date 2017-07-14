import numpy as np 
import cv2

detector = cv2.CascadeClassifier('/home/geekysethi/opencv-3.2.0/data/haarcascades/haarcascade_fullbody.xml')

img = cv2.imread('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09 (3)/S0/City_Center/Time_12-34/View_001/frame_0000.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pedestrian = detector.detectMultiScale(gray, 1.05, 3)

print(pedestrian)

for (x,y,w,h) in pedestrian:
	cv2.rectangle(img,(x-25,y-25),(x+w+25,y+h+25),(255,0,0),2)

cv2.imshow('dfd',img)
cv2.waitKey(0)
cv2.destroyAllWindows()