import cv2 
import numpy as np 
import glob 
from matplotlib import pyplot as plt 
import backgroundModelFunctions
import backgroundModel


# path of images
path = np.sort(glob.glob('Images/test*.bmp'))
# first frame for Height and width
firstFrame = cv2.imread(path[0],0)
bgModel=backgroundModel.backgroundModel(firstFrame,1,2)
kernel = np.ones((3,3),np.uint8)



'''
alpha=1
for i in path[0:100]:
	currentImage=cv2.imread(i,0)
	binaryImage=backgroundModelFunctions.change_detection(firstFrame,currentImage,15)
	bgModel.update_alpha(alpha)
	bgModel.updateBackgroundModel(currentImage,binaryImage)
	print(bgModel.alpha)
	alpha+=1
	print(i)
	plt.subplot(2,2,1)
	plt.imshow(currentImage,cmap='gray')
	plt.subplot(2,2,2)
	plt.imshow(binaryImage,cmap='gray')
	plt.subplot(2,2,3)
	plt.imshow(bgModel.meanImage,cmap='gray')
	plt.subplot(2,2,4)
	plt.imshow(bgModel.varianceImage,cmap='gray')
	plt.pause(0.05)

cv2.imshow("current Image",binaryImage)
cv2.imshow('meanImage',bgModel.meanImage)
cv2.imshow('varianceImage',bgModel.varianceImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
print('='*80)

count=1
bgModel.update_alpha(2)

for j in path[101:]:
	print(j)
	currentImage = cv2.imread(j,0)
	binaryImage=backgroundModelFunctions.change_detection(firstFrame,currentImage,15)
	bgModel.updateBackgroundModel(currentImage,binaryImage)
	

	plt.subplot(2,2,1)
	plt.imshow(currentImage,cmap='gray')
	plt.subplot(2,2,2)
	plt.imshow(binaryImage,cmap='gray')
	plt.subplot(2,2,3)
	plt.imshow(bgModel.meanImage,cmap='gray')
	plt.subplot(2,2,4)
	plt.imshow(bgModel.varianceImage,cmap='gray')
	plt.pause(0.05)


while(True):

	plt.pause(0.05)
	