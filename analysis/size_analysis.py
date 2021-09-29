"""
code to compare 2 models with precision, recall and F1 score over a set of images.
"""
import tensorflow as tf
from skimage.io import imread
import os
import numpy as np
import random
import cv2
import imutils
from PIL import Image
import matplotlib.pyplot as plt


modelBest = tf.keras.models.load_model(
    "modelo_best_patches_100_40.hdf5",compile=False)
modelModelo11 = tf.keras.models.load_model(
    "modelo11_50_10_midRes.hdf5", compile=False)

SIZE =256

path_images = "D:\\UNet\\vegetation\\images\\imagesGrayPatches\\"
path_masks = "D:\\UNet\\vegetation\\masks\\masksPatches\\"

seed=22

images = os.listdir(path_images)
masks = os.listdir(path_masks)
random.shuffle(images)

sizes = np.linspace(0,240,17)
quantities_masks = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0, 0, 0, 0]
quantities_best = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0, 0, 0, 0]
quantities_model11 = [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0, 0, 0, 0, 0, 0, 0]

def calc_predictions(image):

    image_norm = np.expand_dims(tf.keras.utils.normalize(np.array(image), axis=1, order = 2),2)
    image_norm = image_norm[:,:,0][:,:,None]
    image_input = np.expand_dims(image_norm,0)*10

    predictions = np.zeros((SIZE,SIZE,2))

    predictions[:,:,0] = (modelBest.predict(image_input)[0,:,:,0] > 0.15).astype(np.uint8)
    predictions[:,:,1] = (modelModelo11.predict(image_input)[0,:,:,0] > 0.15).astype(np.uint8)

    return predictions

def getCountours(image):

    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    return contours

def getDetectionsParamsFromMasks(quantities_masks,mask):

    cnts = getCountours(mask.copy())

    centroids = np.empty([3,len(cnts)])

    i = 0
    for c in cnts:

        # compute the center of the contour
    
        M = cv2.moments(c)

        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = cv2.contourArea(c)

        centroids[0][i] = cX
        centroids[1][i] = cY
        centroids[2][i] = area

        for z in range(0, len(sizes)-1):
                    if area > sizes[z] and area < sizes[z+1]:
                        quantities_masks[z] += 1

        i += 1
    return quantities_masks,centroids


processed_images = 0

for image in images:
    #ensuring that the file is png type
    if image.split('.')[1] != "png":
        continue

    mask = imread(path_masks + image)
    mask = Image.fromarray(mask)
    mask = mask.resize((SIZE,SIZE))
    mask = np.array(mask)

    #getting shapes from masks
    quantities_masks, centroids = getDetectionsParamsFromMasks(quantities_masks, mask)


    #computing predictions
    file = cv2.imread(path_images + image)[:,:,0]
    file = cv2.resize(file,(SIZE,SIZE), cv2.INTER_LINEAR)
    predictions = calc_predictions(file)

    #getting predictions contours
    pred_cnts_best = getCountours(predictions[:,:,0].astype(np.uint8))

    pred_cnts_model11 = getCountours(predictions[:,:,1].astype(np.uint8))

    #calculates number of missed objects in predictions for each model and size
    for i in range(0, len(centroids[0])):
        
        
        point = (centroids[0][i], centroids[1][i])

        #model best
        pred_detection = 0

        for c in pred_cnts_best:
            
            if cv2.pointPolygonTest(c, point, False) == 1:

                pred_detection = 1

        if pred_detection == 0:

            for z in range(0, len(sizes)-1):
                if centroids[2][i] > sizes[z] and centroids[2][i] < sizes[z+1]:
                    quantities_best[z] += 1
        #model 11
        pred_detection = 0

        for c in pred_cnts_model11:

            if cv2.pointPolygonTest(c, point, False) == 1:

                pred_detection = 1

        if pred_detection == 0:

            for z in range(0, len(sizes)-1):
                if centroids[2][i] > sizes[z] and centroids[2][i] < sizes[z+1]:
                    quantities_model11[z] += 1
    
    processed_images += 1
    print(processed_images)

print(sizes)
print(quantities_masks)
print(np.sum(np.array(quantities_masks)))
print(quantities_best)
print(np.sum(np.array(quantities_best)))
print(quantities_model11)
print(np.sum(np.array(quantities_model11)))


ymin = 0
ymax = 3600

plt.figure(figsize = (16,8))
ax1 = plt.subplot(221)
plt.title("mascaras")
plt.bar(sizes[1:], quantities_masks,3)
ax1.set_ylim(ymin, ymax)
ax2 = plt.subplot(223)
plt.title("model best")
plt.bar(sizes[1:], quantities_best,3)
ax2.set_ylim(ymin, ymax)
ax3 = plt.subplot(224)
plt.title("model 11")
plt.bar(sizes[1:], quantities_model11,3)
ax3.set_ylim(ymin, ymax)
plt.show()

   