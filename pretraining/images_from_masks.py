#This code pretends to collect all images from
#raw folders using their mask names dumping them in only one folder.
import os
import shutil


xavi_path = "D:\\UNet\\vegetation\\grayXavi\\"
javi_path = "D:\\UNet\\vegetation\\grayJavi\\"
hector_path = "D:\\UNet\\vegetation\\grayHector\\"
dest_path = "D:\\UNet\\vegetation\\images\\"
masks_path = "D:\\UNet\\vegetation\\masksBinary\\"




masks = os.listdir(masks_path)
images_javi = os.listdir(javi_path)
images_xavi = os.listdir(xavi_path)
images_hector = os.listdir(hector_path)

for mask in masks:

	if mask in images_javi:
		shutil.copyfile(javi_path + mask, dest_path + mask)
	if mask in images_xavi:
		shutil.copyfile(xavi_path + mask, dest_path + mask)
	if mask in images_hector:
		shutil.copyfile(hector_path + mask, dest_path + mask)