"""
This code visually shows the predictions of a model over images without labels.
"""
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from skimage.io import imread
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


model = tf.keras.models.load_model("best.hdf5", compile=False)

SIZE =256

path = "/Users/javier/Desktop/UNet/vegetation/SUDAN/cultivos/gray/"

seed=22

images = os.listdir(path)

for image in images:
    #image = imread(path + image)[:,:,0]
    image = imread(path + image)
    #image = image/255.

    image2 = Image.fromarray(image)
    image2 = image2.resize((SIZE,SIZE))

    image_norm = np.expand_dims(tf.keras.utils.normalize(np.array(image2), axis=1, order = 2),2)
    image_norm = image_norm[:,:,0][:,:,None]
    image_input = np.expand_dims(image_norm,0)*10
    
    prediction = (model.predict(image_input)[0,:,:,0] > 0.8).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(211)
    plt.title('image')
    plt.imshow(image2, cmap = 'gray')
    plt.subplot(212)
    plt.title('prediction')
    plt.imshow(prediction, cmap = 'gray')
    plt.show()