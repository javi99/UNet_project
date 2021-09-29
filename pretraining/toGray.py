"""
This code only loads images of ndvi and saves only 1 channel (gray)
"""
import cv2
import os
import numpy as np 

import sys

args = sys.argv

path = args[1]
dest_path = args[2]

images = os.listdir(path)

for image in images:

	if image.split('.')[1] != "png":
		continue

	image2 = cv2.imread(path + image)
	blue = image2[:,:,0]
	
	#blue = cv2.cvtColor(blue,cv2.COLOR_GRAY2BGR)

	#hori = np.concatenate((image2,blue), axis = 1)

	#cv2.imshow("hori", blue)
	#cv2.waitKey(0)
	#gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

	cv2.imwrite(dest_path + image,blue)