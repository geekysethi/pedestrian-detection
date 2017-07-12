import cv2 
import numpy as np 
import glob 
from imutils import contours
from skimage import measure
import imutils



# path = np.sort(glob.glob('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09/S0/Background/View_001/Time_13-06/*.jpg'))
path1 =np.sort(glob.glob('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09 (3)/S0/City_Center/Time_12-34/View_001/*.jpg'))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# frame = cv2.imread('/home/geekysethi/Desktop/pedestrain-detection/demo.jpg',1)
# # cv2.imshow('image',frame)

# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (11, 11), 0)
# thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

# thresh = cv2.erode(thresh, None, iterations=2)
# thresh = cv2.dilate(thresh, None, iterations=4)

# # cv2.imshow('fd',thresh)


# labels = measure.label(thresh, neighbors=8, background=0)
# mask = np.zeros(thresh.shape, dtype="uint8")


# for label in np.unique(labels):
# 	# if this is the background label, ignore it
# 	if label == 0:
# 		continue
 
# 	# otherwise, construct the label mask and count the
# 	# number of pixels 
# 	labelMask = np.zeros(thresh.shape, dtype="uint8")
# 	labelMask[labels == label] = 255
# 	numPixels = cv2.countNonZero(labelMask)
 
# 	# if the number of pixels in the component is sufficiently
# 	# large, then add it to our mask of "large blobs"
# 	if numPixels > 300:
# 		mask = cv2.add(mask, labelMask)

# # cv2.imshow('mask',mask)

# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)

# cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# # cnts = contours.sort_contours(cnts)[0]
# # print(cnts[0])
# # print("="*80)
# # print(cnts[1])

# for (i, c) in enumerate(cnts):
# 	# draw the bright spot on the image
# 	# print(c)
# 	print('============')
# 	print(cv2.boundingRect(c))
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)
# 	# ((cX, cY), radius) = cv2.minEnclosingCircle(c)

# # show the output image
# cv2.imshow("Image", frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

for i in path1:

	frame = cv2.imread(i,1)
	fgmask = fgbg.apply(frame)
	cv2.imshow('frame',fgmask)
	cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	for (i, c) in enumerate(cnts):
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)
	
	cv2.imshow("Image", frame)





	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cv2.destroyAllWindows()


