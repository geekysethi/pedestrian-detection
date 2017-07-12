import cv2 
import numpy as np 
import glob 
from imutils import contours
from skimage import measure
import imutils



# path = np.sort(glob.glob('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09/S0/Background/View_001/Time_13-06/*.jpg'))
path1 =np.sort(glob.glob('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09 (3)/S0/City_Center/Time_12-34/View_001/*.jpg'))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
kernel = np.ones((3,3),np.uint8)


for i in path1:

	frame = cv2.imread(i,1)
	fgmask = fgbg.apply(frame)
	cv2.imshow('frame',fgmask)

	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	cv2.imshow('frafgdme',fgmask)

	labels = measure.label(fgmask, neighbors=8, background=0)
	mask = np.zeros(fgmask.shape, dtype="uint8")


	for label in np.unique(labels):

		if label == 0:
			continue
 

		labelMask = np.zeros(fgmask.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
 
		if numPixels > 300:
			mask = cv2.add(mask, labelMask)

	cv2.imshow('mask',mask)





	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# print(cnts(0))
	for (i, c) in enumerate(cnts):
		(x, y, w, h) = cv2.boundingRect(c)

		(rects, weights) = hog.detectMultiScale(frame[y:y+h,x:x+w], winStride=(4, 4),padding=(8, 8), scale=1.05)

		print(i)

		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)
	
	cv2.imshow("Image", frame)





	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cv2.waitKey(0)
cv2.destroyAllWindows()


