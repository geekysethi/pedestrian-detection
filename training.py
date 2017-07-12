import cv2 
import numpy as np 
import glob 
from matplotlib import pyplot as plt 
import backgroundModelFunctions
import backgroundModel
# import tqdm
import _pickle as cpickle
import time
# np.set_printoptions(threshold=np.inf)


path = np.sort(glob.glob('/home/geekysethi/Desktop/pedestrain-detection/Crowd_PETS09/S0/Background/View_001/Time_13-06/*.jpg'))



firstFrame = cv2.imread(path[0],0)
bgModel=backgroundModel.backgroundModel(firstFrame,1,.7)
kernel = np.ones((3,3),np.uint8)


start_time=time.clock()
alpha=1
count=0
for i in path[20:120]:
	firstFrame = cv2.imread(path[count],0)
	count+=1
	currentImage=cv2.imread(i,0)
	binaryImage=backgroundModelFunctions.change_detection(firstFrame,currentImage,15)
	bgModel.update_alpha(alpha)
	bgModel.updateBackgroundModel(currentImage,binaryImage)
	print(bgModel.alpha)
	alpha+=1
	print(i)

	# plt.subplot(2,2,1)
	# plt.imshow(currentImage,cmap='gray')
	# plt.subplot(2,2,2)
	# plt.imshow(binaryImage,cmap='gray')
	# plt.subplot(2,2,3)
	# plt.imshow(bgModel.meanImage,cmap='gray')
	# plt.subplot(2,2,4)
	# plt.imshow(bgModel.varianceImage,cmap='gray')
	# plt.pause(0.005)

print('Total time: '+str(time.clock()-start_time))

cv2.imshow("current Image",binaryImage)
cv2.imshow('meanImage',bgModel.meanImage)
cv2.imshow('varianceImage',bgModel.varianceImage)
print('='*80)
print("Dumping Data")
pickle_out1=open('meanImage.pickle','wb')
cpickle.dump(bgModel.meanImage,pickle_out1)

pickle_out2=open('VarianceImage.pickle','wb')
cpickle.dump(bgModel.varianceImage,pickle_out2)
print('Data Dumped')
print('='*80)


cv2.waitKey(0)
cv2.destroyAllWindows()



