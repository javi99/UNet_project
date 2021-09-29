"""
code to separate labeled images in training set, dev set, test set.
"""
import splitfolders

input_folder = '/Users/javier/Desktop/UNet/vegetation/datasets/data'

splitfolders.ratio(input_folder, output="/Users/javier/Desktop/UNet/vegetation/datasets/dataGray2", seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values

