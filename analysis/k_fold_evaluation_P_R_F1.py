from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
import cv2



def PrecRecF1(mask, prediction):

    TP = np.sum(np.logical_and(mask, prediction))
    FN = np.sum(np.logical_and(mask, (~prediction.astype(bool)).astype(int)))
    FP = np.sum(np.logical_and((~mask.astype(bool)).astype(int), prediction))

    if TP == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, F1

def calc_prediction(image, model):

    image_norm = np.expand_dims(tf.keras.utils.normalize(np.array(image), axis=1, order = 2),2)
    image_norm = image_norm[:,:,0][:,:,None]
    image_input = np.expand_dims(image_norm,0)*10

    prediction = (model.predict(image_input)[0,:,:,0] > 0.15).astype(np.uint8)

    return prediction


def evaluate_model(model, augmented, seed, dataset_name):
    precision_list = []
    recall_list = []
    F1_list = []    

    if augmented:
        img_data_gen_args = dict(rescale = 1/255.,
                         rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')

        mask_data_gen_args = dict(rescale = 1/255.,  #Original pixel values are 0 and 255. So rescaling to 0 to 1
                        rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect',
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

    if not augmented:
        img_data_gen_args = dict(rescale = 1/255.)

        mask_data_gen_args = dict(rescale = 1/255.,
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again.

    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    test_img_generator = image_data_generator.flow_from_directory(dataset_name + "/test_images/", 
                                                              seed=seed, 
                                                              batch_size=len(os.listdir(dataset_name+ "/test_images/")),
                                                              color_mode = 'grayscale', 
                                                              #target_size = (128,128),
                                                              class_mode=None) #Default batch size 32, if not specified here
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    test_mask_generator = mask_data_generator.flow_from_directory(dataset_name + "/test_masks/", 
                                                              seed=seed, 
                                                              batch_size=len(os.listdir(dataset_name+ "/test_masks/")), 
                                                              color_mode = 'grayscale',   #Read masks in grayscale
                                                              #target_size = (128,128),
                                                              class_mode=None)  #Default batch size 32, if not specified here

    x = test_img_generator.next()
    y = test_mask_generator.next()

    for i in range(len(os.listdir(dataset_name+ "/test_images/"))):

        mask = y[i]
        prediction = calc_prediction(x[i], model)

        precision, recall, F1 = PrecRecF1(mask, prediction)

        precision_list.append(precision)
        recall_list.append(recall)
        F1_list.append(F1)

    mean_precision = np.mean(np.array(precision_list))
    mean_recall = np.mean(np.array(recall_list))
    mean_F1 = np.mean(np.array(F1_list))

    return mean_precision, mean_recall, mean_F1


