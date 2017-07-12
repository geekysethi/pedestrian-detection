
import cv2 
import numpy as np 
import glob 
from matplotlib import pyplot as plt 
import backgroundModelFunctions
import backgroundModel
import tqdm
import _pickle as cpickle
import os 


outputImagePath='/home/ashish/Desktop/summer_IITG/outputImages'
meanImagePath='/home/ashish/Desktop/summer_IITG/MeanImages'
if not os.path.exists(outputImagePath):
    os.makedirs(outputImagePath)

if not os.path.exists(meanImagePath):
    os.makedirs(meanImagePath)

# path of images
path = np.sort(glob.glob('Images/test*.bmp'))
# first frame for Height and width
firstFrame = cv2.imread(path[0],0)
bgModel=backgroundModel.backgroundModel(firstFrame,1,4)
kernel = np.ones((3,3),np.uint8)

# opening saved files of mean image and variance image
pickle_in = open('meanImage1.pickle','rb')
meanImage = cpickle.load(pickle_in)
pickle_in = open('VarinaceImage1.pickle','rb')
varianceImage = cpickle.load(pickle_in)
bgModel.update_meanAndvarianceImage(meanImage,varianceImage)



cv2.imshow('meanImage',bgModel.meanImage)
cv2.imshow('VarianceImage',bgModel.varianceImage)
cv2.waitKey()
cv2.destroyAllWindows()

# setting value of alpha

count=80
bgModel.update_alpha(12)
picNum=1
for j in path[100:]:
	print(j)
	firstFrame = cv2.imread(path[count],0)
	count+=1
	currentImage = cv2.imread(j,0)
	'''
	plt.subplot(2,2,1)
	plt.imshow(currentImage,cmap='gray')
	plt.title('current Image')
	'''
	# background Subtraction taking place
	output =bgModel.backgroundSubtraction(currentImage)
	'''
	plt.subplot(2,2,2)
	plt.imshow(output,cmap='gray')
	plt.title("Frame Number-"+str(picNum))
	'''
	

	output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
	'''
	plt.subplot(2,2,3)
	plt.imshow(output,cmap='gray')
	'''
	binaryImage=backgroundModelFunctions.change_detection(firstFrame,currentImage,15)
	bgModel.updateBackgroundModel(currentImage,binaryImage)
	'''
	plt.subplot(2,2,4)
	plt.imshow(bgModel.meanImage,cmap='gray')
	plt.pause(0.05)
	'''
	cv2.imwrite(outputImagePath+'/'+str(picNum)+".jpg",output)
	cv2.imwrite(meanImagePath+'/'+str(picNum)+".jpg",bgModel.meanImage)
	picNum+=1



while(True):

	plt.pause(0.05)
