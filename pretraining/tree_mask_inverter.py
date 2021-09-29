"""
This code inverts tree labels obtained 
"""
import cv2
import os
import numpy as np

or_path = "SUDAN/arboles/masks_inverted"
dest_path = "SUDAN/arboles/masks"

images = os.listdir(or_path)

for image in images:

	image_file = cv2.imread(os.path.join(or_path,image))
	
	inverted = np.zeros((image_file.shape))
	inverted[image_file == 0] = 255

	
	cv2.imwrite(os.path.join(dest_path, image), inverted)