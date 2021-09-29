"""
code to compare 2 models with precision, recall and F1 score over a set of images.
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

modelBest = tf.keras.models.load_model("/Users/javier/Desktop/UNet/vegetation/final_weights/modelo_best.hdf5", compile=False)
modelModelo11 = tf.keras.models.load_model("/Users/javier/Desktop/UNet/vegetation/final_weights/modelo11_1100_200.hdf5", compile=False)


SIZE =256

path_images = "/Users/javier/Desktop/UNet/vegetation/images/imagesBlueNoTrees/"
path_masks = "/Users/javier/Desktop/UNet/vegetation/masks/masksNoTrees/"

seed=22

F1_best_list = []
F1_model11_list = []
precision_best_list = []
precision_model11_list = []
recall_best_list = []
recall_model11_list = []

malas_pred_best = dict()
malas_pred_modelo11 = dict()

i = 0

images = os.listdir(path_images)
masks = os.listdir(path_masks)
random.shuffle(images)
print(images)

threshold = 0.1


worseImages_bestModelBest = dict()
worseImages_bestModel11 = dict()
bestThresholdBest = 0.1
bestThresholdModel11 = 0.1


def PrecRecF1(mask, predictions):
    TP_best = np.sum(np.logical_and(mask, predictions[:,:,0]))
    TP_model11 = np.sum(np.logical_and(mask, predictions[:,:,1]))

    FN_best = np.sum(np.logical_and(mask, (~predictions[:,:,0].astype(bool)).astype(int)))
    FN_model11 = np.sum(np.logical_and(mask, (~predictions[:,:,1].astype(bool)).astype(int)))

    FP_best = np.sum(np.logical_and((~mask.astype(bool)).astype(int), predictions[:,:,0]))
    FP_model11 = np.sum(np.logical_and((~mask.astype(bool)).astype(int), predictions[:,:,1]))

    precision_best = TP_best / (TP_best + FP_best)
    precision_model11 = TP_model11 / (TP_model11 + FP_model11)

    recall_best = TP_best / (TP_best + FN_best)
    recall_model11 = TP_model11 / (TP_model11 + FN_model11)

    F1_best = 2 * (precision_best * recall_best) / (precision_best + recall_best)
    F1_model11 = 2 * (precision_model11 * recall_model11) / (precision_model11 + recall_model11)

    return precision_best, precision_model11, recall_best, recall_model11, F1_best, F1_model11

def calc_predictions(image):

    image_norm = np.expand_dims(tf.keras.utils.normalize(np.array(image), axis=1, order = 2),2)
    image_norm = image_norm[:,:,0][:,:,None]
    image_input = np.expand_dims(image_norm,0)*10

    predictions = np.zeros((SIZE,SIZE,2))

    predictions[:,:,0] = (modelBest.predict(image_input)[0,:,:,0] > 0.15).astype(np.uint8)
    predictions[:,:,1] = (modelModelo11.predict(image_input)[0,:,:,0] > 0.15).astype(np.uint8)

    return predictions

for image in images:
    #ensuring that the file is png type
    if image.split('.')[1] != "png":
        continue
    image2 = cv2.imread(path_images + image)[:,:,0]
    image2 = cv2.resize(image2,(SIZE,SIZE), cv2.INTER_LINEAR)

    #computing predictions
    predictions = calc_predictions(image2)

    #creating pictures to be painted
    painted = cv2.imread(path_images+image)[:,:,0]
    painted = cv2.resize(painted, (SIZE,SIZE), cv2.INTER_LINEAR)
    painted = cv2.cvtColor(painted,cv2.COLOR_GRAY2BGR)

    painted2 = painted.copy()

    #painting pictures
    painted[predictions[:,:,0]>0] = [0,0,255]
    painted2[predictions[:,:,1]>0] = [255,0,0]

    mask = imread(path_masks + image)[:,:,0]
    mask = Image.fromarray(mask)
    mask = mask.resize((SIZE,SIZE))
    mask = np.array(mask)

    #computing precision, recall, and F1 score
    precision_best, precision_model11, recall_best, recall_model11, F1_best, F1_model11 = PrecRecF1(mask, predictions)

    if F1_best < 0.2:
        malas_pred_best[image] = [precision_best, recall_best, F1_best]

    if F1_model11 < 0.2:
        malas_pred_modelo11[image] = [precision_model11, recall_model11, F1_model11]

    F1_best_list.append(F1_best)
    F1_model11_list.append(F1_model11)
    precision_best_list.append(precision_best)
    precision_model11_list.append(precision_model11)
    recall_best_list.append(recall_best)
    recall_model11_list.append(recall_model11)

    mean_F1_best = np.mean(np.array(F1_best_list))
    mean_F1_model11 = np.mean(np.array(F1_model11_list))
    mean_precision_best = np.mean(np.array(precision_best_list))
    mean_precision_model11 = np.mean(np.array(precision_model11_list))
    mean_recall_best = np.mean(np.array(recall_best_list))
    mean_recall_model11 = np.mean(np.array(recall_model11_list))

    print(f"""\n model best; 
        F1: {F1_best:.2f},
        Precision: {precision_best:.2f},
        Recall: {recall_best:.2f} 
        on image {image} \n""")

    print(f"""mean values model Best;  
        F1: {mean_F1_best:.2f},
        Precision: {mean_precision_best:.2f},
        Recall: {mean_recall_best:.2f} 
        on image {image} \n""")     

    print(f"""model 11; 
        F1: {F1_model11:.2f},
        Precision: {precision_model11:.2f},
        Recall: {recall_model11:.2f} 
        on image {image} \n""")
    
    print(f"""mean values model 11;  
        F1: {mean_F1_model11:.2f},
        Precision: {mean_precision_model11:.2f},
        Recall: {mean_recall_model11:.2f} 
        on image {image} \n""")

    #computing intersections to show visually the performance
    intersectionBest = np.logical_and(mask, predictions[:,:,0])
    intersectionModelo11 = np.logical_and(mask, predictions[:,:,1])

    #showing some examples
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

print("predicciones malas Best: ")
print(malas_pred_best)
print("predicciones malas Modelo11: ")
print(malas_pred_modelo11)

    

