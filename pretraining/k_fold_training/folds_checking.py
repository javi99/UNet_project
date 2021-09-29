#code to ensure that all folds are different
import os

base_path = "/Users/javier/Desktop/UNet/vegetation/datasets/folds"
complementary_path = "test_images/test"

for i in range(1, 11):
	images_fold = os.listdir(os.path.join(base_path, "fold" + str(i), complementary_path))
	images_fold.sort()
	print(images_fold)
	for j in range(1, 11):

		if j == i:
			continue

		images_to_compare = os.listdir(os.path.join(base_path, "fold" + str(j), complementary_path))
		images_to_compare.sort()
		print(images_to_compare)

		if images_to_compare == images_fold:
			print("folds " + str(i) + " and " + str(j) + " are equal")

	print("\n\n")