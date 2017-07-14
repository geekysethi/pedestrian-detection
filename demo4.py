import cv2 
import numpy as np 
import glob 
from imutils import contours
from skimage import measure
import imutils



# path = np.sort(glob.glob('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09/S0/Background/View_001/Time_13-06/*.jpg'))
path1 =np.sort(glob.glob('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09 (3)/S0/City_Center/Time_12-34/View_001/*.jpg'))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

detector = cv2.CascadeClassifier('/home/geekysethi/opencv-3.2.0/data/haarcascades/haarcascade_fullbody.xml')

kernel = np.ones((3,3),np.uint8)

count=1
for i in path1[0:250]:
	print("="*10,'FRAME NO.',count,'='*10)
	count+=1
	print("="*40)
	frame = cv2.imread(i,1)
	fgmask = fgbg.apply(frame)
	cv2.imshow('frame',fgmask)

	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	cv2.imshow('frafgdme',fgmask)

	labels = measure.label(fgmask, neighbors=4, background=0)
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
		print(y,'',y+h,'',x,'',x+w)
		# ROI = frame[64+y:y+h+64,+64+x:64+x+w]
		
		ROI = frame[y-50:y+h+50,x-50:x+w+50]
		pedestrian = detector.detectMultiScale(ROI, 1.05, 3)
		# for (xc yc wc hc) in pedestrian:

		print(pedestrian)
		cv2.imshow("ROI",ROI)
		# cv2.waitKey(0)

		print(i)

		# cv2.rectangle(frame, (x-50,y-50), (x+w+50,y+h+50), (0,255,0),3)
	
	cv2.imshow("Image", frame)





	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cv2.waitKey(0)
cv2.destroyAllWindows()


