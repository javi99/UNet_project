"""
code to visually compare performance of 2 models
"""


from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from skimage.io import imread
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random
import cv2

models_list = ["modelo_best.hdf5","modelo11_1100_200.hdf5"]



SIZE =256

path = "/Users/javier/Desktop/UNet/vegetation/test_im_prueba/"

seed=22

images = os.listdir(path)
random.shuffle(images)
print(images)

for image in images:
    if image.split('.')[1] != "png":
        continue
    image2 = cv2.imread(path + image)[:,:,0]
    image2 = cv2.resize(image2,(SIZE,SIZE), cv2.INTER_LINEAR)
    """image2 = imread(path + image)[:,:,2]
                image2 = Image.fromarray(image2)
            
                image2 = image2.resize((SIZE,SIZE))"""

    image_norm = np.expand_dims(tf.keras.utils.normalize(np.array(image2), axis=1, order = 2),2)
    image_norm = image_norm[:,:,0][:,:,None]
    image_input = np.expand_dims(image_norm,0)*10

    predictions = np.zeros((SIZE,SIZE,2))

    for index in range(len(models_list)):

        model = tf.keras.models.load_model(models_list[index], compile=False)
        
        prediction = (model.predict(image_input)[0,:,:,0] > 0.5).astype(np.uint8)

        predictions[:,:,index] = prediction

    painted = cv2.imread(path+image)[:,:,0]
    painted = cv2.resize(painted, (SIZE,SIZE), cv2.INTER_LINEAR)
    painted = cv2.cvtColor(painted,cv2.COLOR_GRAY2BGR)

    painted2 = painted.copy()

    painted[predictions[:,:,0]>0] = [0,0,255]
    painted2[predictions[:,:,1]>0] = [255,0,0]
     
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title(image)
    plt.imshow(image2, cmap = 'gray')
    plt.subplot(232)
    plt.title(image)
    plt.imshow(painted)
    plt.subplot(233)
    plt.title(image)
    plt.imshow(painted2)
    plt.subplot(235)
    plt.title('model ' + models_list[0])
    plt.imshow(predictions[:,:,0], cmap = 'gray')
    plt.subplot(236)
    plt.title('model ' + models_list[1])
    plt.imshow(predictions[:,:,1], cmap = 'gray')
    plt.show()

