import cv2 
import numpy as np 
import glob 
from matplotlib import pyplot as plt 
import backgroundModelFunctions
from itertools import product




class backgroundModel:

	def __init__(self,firstFrame,alpha,K):

		(imgHeight,imgWidth)=firstFrame.shape[:2]
		self.alpha = alpha
		self.K=K
		self.imgHeight=imgHeight
		self.imgWidth=imgWidth
		self.meanImage = np.zeros((imgHeight,imgWidth),dtype=np.float)
		self.varianceImage = np.zeros((imgHeight,imgWidth),dtype=np.float)
		self.output = np.zeros((self.imgHeight,self.imgWidth),dtype = np.float)


	def updateBackgroundModel(self,currentFrame,binaryImage):
		# k=np.where(binaryImage==0)
		for i,j in product(range(self.imgHeight),range(self.imgWidth)):
				
			if(binaryImage[i,j]==0):
				mean = (1-self.alpha)*self.meanImage[i,j]+self.alpha*currentFrame[i,j]

				variance = (1-self.alpha)*self.varianceImage[i,j]+self.alpha*((int(currentFrame[i,j])
					-self.meanImage[i,j]).transpose())*(int(currentFrame[i,j])-self.meanImage[i,j])

				self.meanImage[i,j]=mean
				self.varianceImage[i,j]=variance

		self.meanImage = self.meanImage.astype(np.uint8)
		self.varianceImage = self.varianceImage.astype(np.uint8)
		'''
		self.varianceImage[k] = (1-self.alpha)*self.varianceImage[k]+self.alpha*((currentFrame[k]
			-self.meanImage[k]).transpose())*(currentFrame[k]-self.meanImage[k])

		self.meanImage[k] = (1-self.alpha)*self.meanImage[k]+self.alpha*currentFrame[k]	
		
		self.meanImage.astype(np.uint8)
		self.varianceImage.astype(np.uint8)
		'''	



	def update_meanAndvarianceImage(self,meanImage,varianceImage):

		self.meanImage=meanImage
		self.varianceImage=varianceImage

	def update_alpha(self,alpha):

		self.alpha = 1/(alpha+1)
		
				

	def backgroundSubtraction(self,currentFrame):
		
		self.output = np.ones((self.imgHeight,self.imgWidth),dtype = np.float)
		self.output = 255*self.output
		lowerBoundValues = np.where(currentFrame<=self.meanImage +self.K*np.sqrt(self.varianceImage))
		uperBoundValues = np.where(currentFrame>=self.meanImage -self.K*np.sqrt(self.varianceImage))

		print(type(list(lowerBoundValues)))
		i= lowerBoundValues and uperBoundValues

		self.output[i]=0
		return self.output.astype(np.uint8)







