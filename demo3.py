import cv2
import numpy as np 
import imutils

from imutils.object_detection import non_max_suppression
import glob




hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread('/home/geekysethi/Desktop/summer_IITG/pedestrian_detection/00000001.jpg')
image = imutils.resize(img, width=min(400, img.shape[1]))
orig = image.copy()
 	# detect people in the image
(rects, weights) = hog.detectMultiScale(image[8:8+150,150:150+80], winStride=(4, 4),padding=(8, 8), scale=1.1)
print(weights)
cutted = orig[8:8+150,150:150+80]
# draw the original bounding boxes
for (x, y, w, h) in rects:
	cv2.rectangle(cutted, (x, y), (x + w, y + h), (0, 0, 255), 2)
 

cv2.imshow("Before NMS", cutted)
cv2.imshow("After NMS", image)
cv2.waitKey(0)

















