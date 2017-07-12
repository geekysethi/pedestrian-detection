import numpy as np 
import cv2
from matplotlib import pyplot as plt 
import glob




def denoise(frame):
	
	frame=cv2.medianBlur(frame,5)
	frame=cv2.GaussianBlur(frame,(5,5),0)

	return frame

	


def change_detection(refImage,currentImage,thresh):

	refImage=denoise(refImage)
	currentImage = denoise(currentImage)
	kernel = np.ones((5,5),np.uint8)
	binaryImage = cv2.absdiff(refImage,currentImage)


	th,binaryImage=cv2.threshold(binaryImage,thresh,255,cv2.THRESH_BINARY)
	binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_OPEN, kernel)
	
	return binaryImage

# path = np.sort(glob.glob('Images/test*.bmp'))

# firsFrame = cv2.imread(path[0],0)
# img = cv2.imread(path[150],0)

# output = change_detection(firsFrame,img,15)

# cv2.imshow('image',output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()













