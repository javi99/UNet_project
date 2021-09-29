import splitfolders
import os
import shutil
import cv2

input_folder = 'D:\\UNet\\vegetation\\datasets\\data\\'

#

base_path = "D:\\UNet\\vegetation\\datasets\\folds\\"

number_of_folds = 10

intermediate_paths = ["train", "val","test"]
final_paths = ["images","masks"]

for i in range(1,number_of_folds+1):

	#creates dataset folder well formated
	fold_folder = os.path.join(base_path, "fold" + str(i))

	for final_path in final_paths:
		for intermediate_path in intermediate_paths:

			set_path = os.path.join(fold_folder, intermediate_path + "_" + final_path)
			complete_path = os.path.join(set_path, intermediate_path)

			os.makedirs(complete_path)


	splitfolders.ratio(input_folder, output= os.path.join(base_path, "auxiliary"), seed=1337+i, ratio=(.8, .1, .1), group_prefix=None) 
	

	for intermediate_path in intermediate_paths:
		for final_path in final_paths:

			set_path = os.path.join(base_path,"auxiliary", intermediate_path)
			images_path = os.path.join(set_path, final_path)

			set_path = os.path.join(fold_folder, intermediate_path + "_" + final_path)
			complete_path = os.path.join(set_path, intermediate_path)

			images = os.listdir(images_path)

			for image in images:
				image_origin = os.path.join(images_path, image)
				image_dest = os.path.join(complete_path, image)
				shutil.copyfile(image_origin, image_dest)

	shutil.rmtree(os.path.join(base_path, "auxiliary"))
				
	print("Fold " + str(i) + " done")
			



	
