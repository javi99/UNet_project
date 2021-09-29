# https://youtu.be/csFGTLT6_WQ
"""
Author: Dr. Sreenivas Bhattiprolu
Training and testing for semantic segmentation (Unet) of veg
Uses standard Unet framework with no tricks!
Dataset info: Electron microscopy (EM) dataset from
https://www.epfl.ch/labs/cvlab/data/data-em/
Patches of 256x256 from images and labels 
have been extracted (via separate program) and saved to disk. 
This code uses 256x256 images/masks.
To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""

from simple_unet_model import simple_unet_model   #Use normal unet model
import tensorflow as tf
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize

images_directory = "D:\\UNet\\vegetation\\images\\"
masks_directory = 'D:\\UNet\\vegetation\\masksBinary\\'


SIZE_Y = 512
SIZE_X = 512

image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(images_directory)
for image in images:
    image = imread(images_directory + image)
    image = Image.fromarray(image)
    image = image.resize((SIZE_X,SIZE_Y))
    image_dataset.append(np.array(image))

masks = os.listdir(masks_directory)
for mask in masks:
    mask = imread(masks_directory + mask)
    mask = Image.fromarray(mask)
    mask = mask.resize((SIZE_X,SIZE_Y)) 
    mask_dataset.append(np.array(mask))

image_dataset_uint8 = np.array(image_dataset)
mask_dataset_uint8 = np.array(mask_dataset)

print("used memory to store the 8 bit int image dataset is: ", image_dataset_uint8.nbytes/(1024*1024), "MB")
print("used memory to store the 8 bit int mask dataset is: ", mask_dataset_uint8.nbytes/(1024*1024), "MB")



#Normalize images
image_dataset = np.expand_dims(tf.keras.utils.normalize(np.array(image_dataset), axis=1),3)
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.
print(image_dataset.shape)
print(mask_dataset.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)
print(X_test.shape)
#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))


"""
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (SIZE_X, SIZE_Y)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (SIZE_X, SIZE_Y)), cmap='gray')
plt.show()
"""
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number], cmap='gray')
plt.subplot(122)
plt.imshow(y_train[image_number], cmap='gray')
plt.show()
###############################################################

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()



#If starting with pre-trained weights. 
#model.load_weights('veg_gpu_tf1.4.hdf5')

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=75, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save('veg_test.hdf5')

############################################################
#Evaluate the model


    # evaluate model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#acc = history.history['acc']
acc = history.history['accuracy']
#val_acc = history.history['val_acc']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##################################
#IOU
model = get_model()
model.load_weights('veg_test.hdf5')

y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.2

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#######################################################################
#Predict on a few images
model = get_model()
model.load_weights('veg_test.hdf5') #Trained for 50 epochs and then additional 100
#model.load_weights('veg_gpu_tf1.4.hdf5')  #Trained for 50 epochs

test_img_number = random.randint(0, 1)
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)


test_img_other = imread(images_directory + "_1_0_0.png")
test_img_other = Image.fromarray(test_img_other)
test_img_other = test_img_other.resize((SIZE_X,SIZE_Y))
#test_img_other = cv2.imread('data/test_images/img8.tif', 0)
test_img_other_norm = np.expand_dims(tf.keras.utils.normalize(np.array(test_img_other), axis=1),2)
test_img_other_norm=test_img_other_norm[:,:,0][:,:,None]
test_img_other_input=np.expand_dims(test_img_other_norm, 0)

#Predict and threshold for values above 0.5 probability
#Change the probability threshold to low value (e.g. 0.05) for watershed demo.
prediction_other = (model.predict(test_img_other_input)[0,:,:,0] > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('External Image')
plt.imshow(test_img_other, cmap='gray')
plt.subplot(235)
plt.title('Prediction of external Image')
plt.imshow(prediction_other, cmap='gray')
plt.show()

#plt.imsave('input.jpg', test_img[:,:,0], cmap='gray')
#plt.imsave('data/results/output2.jpg', prediction_other, cmap='gray')