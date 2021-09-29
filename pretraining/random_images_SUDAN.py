"""
code to select x random images of all images from SUDAN
"""

import os

number_of_images = 40
path = "/Users/javier/Desktop/SUDAN/"

images_ref = []
images = dict()

for root, dirs, files in os.walk(path, topdown = "False"):
	for directory in dirs:
		if directory == "raw_images":
			info = os.path.join(root, directory).split('/')
			
			date_raw = info[6]
			
			substring = date_raw[date_raw.find("H"):]
			date = date_raw.replace(date_raw[date_raw.find("H"):],"")

			for img in files:

				