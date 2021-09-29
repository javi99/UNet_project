"""
This code shows predictions over images that have labels, to visually compare
performance of the model. It also shows intersection of prediction and label.
"""
from simple_unet_model import simple_unet_model 
from matplotlib import pyplot as plt
from skimage.io import imread
from PIL import Image
import numpy as np
import tensorflow as tf
import os

SIZE = 512

def get_model():
    return simple_unet_model(SIZE, SIZE, 1)

model = get_model()
model.load_weights('veg_test.hdf5')

path = '/Users/javier/Desktop/UNet/vegetation/images/'
masks = '/Users/javier/Desktop/UNet/vegetation/binary_masks/'
images = os.listdir(path)



for image in images:

    image2 = imread(path + image)
    image2 = Image.fromarray(image2)
    image2 = image2.resize((SIZE,SIZE))

    image_norm = np.expand_dims(tf.keras.utils.normalize(np.array(image2), axis=1),2)
    image_norm = image_norm[:,:,0][:,:,None]
    image_input = np.expand_dims(image_norm,0)
    
    prediction = (model.predict(image_input)[0,:,:,0] > 0.2).astype(np.uint8)

    labels = imread(masks + image)
    labels = Image.fromarray(labels)
    labels = labels.resize((SIZE,SIZE))

    intersection = np.logical_and(labels, prediction)
    union = np.logical_or(labels, prediction)

    plt.figure(figsize=(16, 8))
    plt.subplot(221)
    plt.title('image')
    plt.imshow(image2, cmap = 'gray')
    plt.subplot(222)
    plt.title('intersection')
    plt.imshow(intersection, cmap = 'gray')
    plt.subplot(223)
    plt.title('labels')
    plt.imshow(labels, cmap = 'gray')
    plt.subplot(224)
    plt.title("prediction")
    plt.imshow(prediction, cmap = 'gray')
    plt.show()

    
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU socre is: ", iou_score)
    

