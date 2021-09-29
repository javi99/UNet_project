"""
code to compare 2 models with mean IoU over a set of images.
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

modelBest = tf.keras.models.load_model("modelo_best.hdf5", compile=False)
modelModelo11 = tf.keras.models.load_model("modelo11_1100_200.hdf5", compile=False)


SIZE =256

path_images = "/Users/javier/Desktop/UNet/vegetation/imagesBlueNoTrees/"
path_masks = "/Users/javier/Desktop/UNet/vegetation/masksNoTrees/"

seed=22

IoU_best = []
IoU_modelo11 = []

malas_pred_best = dict()
malas_pred_modelo11 = dict()

i = 0

images = os.listdir(path_images)
masks = os.listdir(path_masks)
random.shuffle(images)
print(images)

threshold = 0.1

bestIoU_bestModel = 0
bestIoU_model11 = 0
worseImages_bestModelBest = dict()
worseImages_bestModel11 = dict()
bestThresholdBest = 0.1
bestThresholdModel11 = 0.1



for image in images:
    if image.split('.')[1] != "png":
        continue
    image2 = cv2.imread(path_images + image)[:,:,0]
    image2 = cv2.resize(image2,(SIZE,SIZE), cv2.INTER_LINEAR)

    image_norm = np.expand_dims(tf.keras.utils.normalize(np.array(image2), axis=1, order = 2),2)
    image_norm = image_norm[:,:,0][:,:,None]
    image_input = np.expand_dims(image_norm,0)*10

    predictions = np.zeros((SIZE,SIZE,2))

    predictions[:,:,0] = (modelBest.predict(image_input)[0,:,:,0] > 0.15).astype(np.uint8)
    predictions[:,:,1] = (modelModelo11.predict(image_input)[0,:,:,0] > 0.15).astype(np.uint8)

    painted = cv2.imread(path_images+image)[:,:,0]
    painted = cv2.resize(painted, (SIZE,SIZE), cv2.INTER_LINEAR)
    painted = cv2.cvtColor(painted,cv2.COLOR_GRAY2BGR)

    painted2 = painted.copy()

    painted[predictions[:,:,0]>0] = [0,0,255]
    painted2[predictions[:,:,1]>0] = [255,0,0]

    mask = imread(path_masks + image)[:,:,0]
    mask = Image.fromarray(mask)
    mask = mask.resize((SIZE,SIZE))

    intersectionBest = np.logical_and(mask, predictions[:,:,0])
    intersectionModelo11 = np.logical_and(mask, predictions[:,:,1])

    unionBest = np.logical_or(mask, predictions[:,:,0])
    unionModelo11 = np.logical_or(mask, predictions[:,:,1])
        
    iouBest = np.sum(intersectionBest)/np.sum(unionBest)
    iouModelo11 = np.sum(intersectionModelo11)/np.sum(unionModelo11)

    if iouBest < 0.2:
        malas_pred_best[image] = iouBest
    if iouModelo11 < 0.2:
        malas_pred_modelo11[image] = iouModelo11

    IoU_best.append(iouBest)
    IoU_modelo11.append(iouModelo11)

    print("model Best" + " IoU: " + str(iouBest) + " on image " + image)
    print(np.mean(np.array(IoU_best)))
    print("modelo 11" + " IoU: " + str(iouModelo11) + " on image " + image)
    print(np.mean(np.array(IoU_modelo11)))

    if i <= 5:
                        
        plt.figure(figsize=(16, 8))
        plt.subplot(331)
        plt.title(image)
        plt.imshow(image2, cmap = 'gray')
        plt.subplot(332)
        plt.title(image)
        plt.imshow(painted)
        plt.subplot(333)
        plt.title(image)
        plt.imshow(painted2)
        plt.subplot(334)
        plt.title(image)
        plt.imshow(mask, cmap = 'gray')
        plt.subplot(335)
        plt.title('model ' + "Best")
        plt.imshow(predictions[:,:,0], cmap = 'gray')
        plt.subplot(336)
        plt.title('model ' + "11")
        plt.imshow(predictions[:,:,1], cmap = 'gray')
        plt.subplot(338)
        plt.title('intersection model ' + "Best")
        plt.imshow(intersectionBest, cmap = 'gray')
        plt.subplot(339)
        plt.title('intersection model ' + "11")
        plt.imshow(intersectionModelo11, cmap = 'gray')
        plt.show()  
    i += 1

    meanBestIoU = np.mean(np.array(IoU_best))
    meanModel11IoU = np.mean(np.array(IoU_modelo11))

print("Mean IoU for Best: " + str(meanBestIoU))
print("Mean IoU for Modelo11: " + str(meanModel11IoU))
print("predicciones malas Best: ")
print(malas_pred_best)
print("predicciones malas Modelo11: ")
print(malas_pred_modelo11)

    









