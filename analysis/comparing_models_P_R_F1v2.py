"""
code to compare 2 models with precision, recall and F1 score over a set of images.
"""

"""
    NECESITA REVISION. QUIZAS AÑADIR OBJETOS PARA MEJORAR EL CODIGO. INTRODUCIR LA POSIBILIDAD DE PONER MODELOS
    CON DIFERENTES CANALES DE ENTRADA Y DIFERENTES TAMAÑOS DE PREDICCION.
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

path_base = "/Users/javier/Desktop/UNet/vegetation"
path_models = os.path.join(path_base, "final_weights/")
path_images = os.path.join(path_base, "images/imagesBlueNoTrees/")
path_masks = os.path.join(path_base, "masks/masksNoTrees/")

list_models = ["modelo_best", "modelo11_1100_200", "modelo4", "modelo9"]

models = dict()
models_shapes = dict()
data_models = dict()

"""
data_models structure:
    model1:
        precision list = [value1, value2, ....]
        recall list = [value1, value2, ....]
        F1 list = [value1, value2, ....]
        bad_pred:
            image1 with low F1 = [F1, precision, recall]
            image2 with low F1 = [F1, precision, recall]
            ...
        num_bad_pred = number of bad predicted images
        mean precision = mean of values of list
        mean recall = mean of values of list
        mean F1 = mean of values of list
    model2:
    ...
    ...
"""

for i in range(len(list_models)):
    models[list_models[i]] = tf.keras.models.load_model(os.path.join(path_models, list_models[i] + ".hdf5"), compile=False)
    models_shapes[list_models[i]] = models[list_models[i]].get_config()
    or_shape = models_shapes[list_models[i]]["layers"][0]["config"]["batch_input_shape"][1:]
exit()

SIZE =256
seed=22

images = os.listdir(path_images)
masks = os.listdir(path_masks)
random.shuffle(images)

def fill_data_models(data_models, list_models, predictions, image, mask):

    #we create the data structure
    for i in range(len(list_models)):

        data_models[list_models[i]] = dict()
        
        data_models[list_models[i]]["precision_list"] = list()
        data_models[list_models[i]]["recall_list"] = list()
        data_models[list_models[i]]["F1_list"] = list()
        data_models[list_models[i]]["bad_pred"] = dict()
        data_models[list_models[i]]["num_bad_pred"] = 0
    #we fill the dictionary with necessary data
    for i in range(len(list_models)):
        
        #computing precision, recall, F1 for each model
        precision, recall, F1 = PrecRecF1(mask, predictions[:,:,i])

        data_models[list_models[i]]["precision_list"].append(precision)
        data_models[list_models[i]]["recall_list"].append(recall)
        data_models[list_models[i]]["F1_list"].append(F1)

        data_models[list_models[i]]["mean_precision"] = \
            np.mean(np.array(data_models[list_models[i] ]["precision_list"]))
        data_models[list_models[i]]["mean_recall"] = \
            np.mean(np.array(data_models[list_models[i] ]["recall_list"]))
        data_models[list_models[i]]["mean_F1"] = \
            np.mean(np.array(data_models[list_models[i] ]["F1_list"]))

        #to keep track of the bad predictions of the model
        if F1 < 0.2:
            data_models[list_models[i] ]["bad_pred"][image] = [F1, precision, recall]
            data_models[list_models[i] ]["num_bad_pred"] += 1

    return data_models

def PrecRecF1(mask, prediction):

    TP = np.sum(np.logical_and(mask, prediction))
    FN = np.sum(np.logical_and(mask, (~prediction.astype(bool)).astype(int)))
    FP = np.sum(np.logical_and((~mask.astype(bool)).astype(int), prediction))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, F1


def calc_predictions(image, list_models, models):

    image_norm = np.expand_dims(tf.keras.utils.normalize(np.array(image), axis=1, order = 2),2)
    image_norm = image_norm[:,:,0][:,:,None]
    image_input = np.expand_dims(image_norm,0)*10

    predictions = np.zeros((SIZE,SIZE, len(list_models)))

    for i in range(len(list_models)):
        model = models[list_models[i] ]
        predictions[:,:,i] = (model.predict(image_input)[0,:,:,0] > 0.15).astype(np.uint8)
    
    return predictions

def painted_images_creator(list_models, predictions):
    painted_images = dict()
    for i in range(len(list_models)):
        painted_image = cv2.imread(path_images+image)[:,:,0]
        painted_image = cv2.resize(painted_image, (SIZE,SIZE), cv2.INTER_LINEAR)
        painted_image = cv2.cvtColor(painted_image,cv2.COLOR_GRAY2BGR)

        painted_image[predictions[:,:,i] > 0] = [0,0,255]

        painted_images[list_models[i]] = painted_image

    return painted_images

def create_intersections(list_models, mask, predictions):
    intersections = dict()
    for i in range(len(list_models)):

        intersections[list_models[i]] = np.logical_and(mask, predictions[:,:,i])

    return intersections

def ploting_results(image, file, mask, painted_images, predictions, intersections, list_models):

    number_of_models = len(list_models)
    indexes = dict()

    plt.figure(figsize=(16, 8))
    indexes["image"] = int(str(3) + str(number_of_models+1) + str(1))
    plt.subplot(indexes["image"])
    plt.title(image)
    plt.imshow(file, cmap = 'gray')
    
    indexes["mask"] = int(str(3) + str(number_of_models+1) + str(number_of_models+2))
    plt.subplot(indexes["mask"])
    plt.title("mask")
    plt.imshow(mask, cmap = 'gray')

    for i in range(3):
        print(i)
        for j in range(1,number_of_models+2):
            print(j)
            indexes[str(10*i + j)] = int(str(3) + str(number_of_models+1) + str(i*(number_of_models+1) + j))
            print(indexes[str(10*i + j)])

            if j == 1:
                continue

            if i == 0:
                plt.subplot(indexes[str(10*i + j)])
                plt.title(list_models[j-2])
                plt.imshow(painted_images[list_models[j-2]])

            if i == 1:
                plt.subplot(indexes[str(10*i + j)])
                plt.title("predictions")
                plt.imshow(predictions[:,:,j-2])

            if i == 2:
                plt.subplot(indexes[str(10*i + j)])
                plt.title("intersections")
                plt.imshow(intersections[list_models[j-2]])


def print_info(data_models, list_models):

    for i in range(len(list_models)):

        print(f"""\n{list_models[i]}; 
            F1: {data_models[list_models[i]]["mean_F1"]:.2f},
            Precision: {data_models[list_models[i] ]["mean_precision"]:.2f},
            Recall: {data_models[list_models[i] ]["mean_recall"]:.2f}  
            \n""")

        print("predicciones malas " + list_models[i] + ": \n" + 
            str(data_models[list_models[i] ]["num_bad_pred"]) + "\n" + 
            str(data_models[list_models[i] ]["bad_pred"]) + "\n")

        print("_________________________" + "\n" + 
                "_________________________")

i = 0

for image in images:
    print(i)

    files, masks = loading_images_masks(image, models_shapes)
    #ensuring that the file is png type and getting image and mask
    if image.split('.')[1] != "png":
        continue
    file = cv2.imread(path_images + image)[:,:,0]
    file = cv2.resize(file,(SIZE,SIZE), cv2.INTER_LINEAR)

    mask = imread(path_masks + image)[:,:,0]
    mask = Image.fromarray(mask)
    mask = mask.resize((SIZE,SIZE))
    mask = np.array(mask)

    #computing predictions
    predictions = calc_predictions(file, list_models, models)

    #creating painted images
    painted_images = painted_images_creator(list_models, predictions)
    
    #filling data dictionary
    data_models = fill_data_models(data_models, list_models, predictions, image, mask)

    #computing intersections to show visually the performance
    intersections = create_intersections(list_models, mask, predictions)
    
    #showing some examples
    if i <= 5:
        ploting_results(image, file, mask, painted_images, predictions, intersections, list_models)
        plt.show()        
    i += 1

print_info(data_models, list_models)
