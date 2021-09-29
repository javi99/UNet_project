"""
This code ensures that labels are binary images (only 0 and 255 values)
"""
import cv2
import os 

path = "/Users/javier/Desktop/UNet/vegetation/masks/"
dest_path = "/Users/javier/Desktop/UNet/vegetation/masksBinary/"

masks = os.listdir(path)

for mask in masks:
	if mask.split('.')[1] == "png":

		image = cv2.imread(path + mask)
		image = image[:,:,0]
		white = image > 0
		image[white] = 255
		cv2.imwrite(dest_path + mask,image)