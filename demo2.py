import cv2 
import numpy as np 
import glob 
from imutils import contours
from skimage import measure
import imutils
import time



# path = np.sort(glob.glob('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09/S0/Background/View_001/Time_13-06/*.jpg'))
path1 =np.sort(glob.glob('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09 (3)/S0/City_Center/Time_12-34/View_001/*.jpg'))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
kernel = np.ones((3,3),np.uint8)

count=1
for i in path1:
	print("="*10,'FRAME NO.',count,'='*10)
	count+=1
	frame = cv2.imread(i,1)
	fgmask = fgbg.apply(frame)
	# cv2.imshow('frame',fgmask)

	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	# cv2.imshow('frafgdme',fgmask)

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
	# print(cnts)
	
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	# for (j, c) in enumerate(cnts):
		# print(j,c)
	
	for (j, c) in enumerate(cnts):
		(x, y, w, h) = cv2.boundingRect(c)
		print(j)
		# print(x, y, x+w,y+h)
		
		# print(abs(y-int(.5*h)),'',y+h+int(.5*h),'',abs(x-w),'',x+int(2*w))
		# ROI = frame[abs(y-int(.5*h)):y+h+int(.5*h),abs(x-w):x+int(2*w)]
		thesh=50
		if(x>thesh and y>thesh):
			temp=np.copy(frame)
			x1coord=x-thesh
			y1coord=y-thesh
			x2coord=x+w+thesh
			y2coord=y+h+thesh
			
			ROI = temp[y1coord:y2coord,x1coord:x2coord]
		# time.sleep(0.1)
		# ROI = frame[int(h*.5)+y:y+h+int(h*.5),w+x:x+2*w]
		
		# cv2.waitKey(0)
		(rects, weights) = hog.detectMultiScale(ROI, winStride=(2, 2),padding=(8, 8), scale=1.1)
		if len(rects)!=0:
			for (xd, yd, wd, hd) in rects:
				print(xd, yd, wd, hd)
			
				xc=x1coord+xd
				yc=y1coord+yd
				cv2.rectangle(frame, (xc,yc), (xc+wd,yc+hd), (0,255,0),2)
		
			# cv2.rectangle(ROI, (newrects[0],newrects[1]), (newrects[0]+newrects[2],newrects[1]+newrects[3]), (0,255,0),3)
			# cv2.imshow("ROI",ROI)

		
		
		# cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)
		
	cv2.imshow("Image", frame)





	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cv2.waitKey(0)
cv2.destroyAllWindows()


