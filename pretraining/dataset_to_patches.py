import cv2
from patchify import patchify, unpatchify
from PIL import Image
import numpy as np
import os

path = "dataGrayNoTrees"

dest = "dataGrayNoTreesPatches"

routes = ["test_images\\test","test_masks\\test",
		  "train_images\\train","train_masks\\train",
		  "val_images\\val","val_masks\\val"]

size = (2048,1536)

for route in routes:

	route_to_dataset = os.path.join(path,route)
	images = os.listdir(route_to_dataset)

	for image in images:

		if image.split(".")[1] != "png":
			continue

		route_to_image = os.path.join(route_to_dataset,image)
		print(route_to_image)
		file = cv2.imread(route_to_image)

		file = cv2.resize(file, size, cv2.INTER_LINEAR)

		patches = patchify(file[:,:,0], (512, 512), step=256)
		print(patches.shape)

		for i in range(patches.shape[0]):
			for j in range(patches.shape[1]):

				dest_path_to_image = os.path.join(dest,route, str(i) + str(j) +image)
				print(dest_path_to_image)
				cv2.imshow("image",patches[i,j,:,:])
				
				
				cv2.imwrite(dest_path_to_image, patches[i,j,:,:])


