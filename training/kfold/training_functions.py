"""This function allows to train a model. It saves it with a determinated name, and returns
the model. The arguments are:
    - epochs: int, number of epochs
    - augmented: boolean, true if data augmentation is desired, false if no modification is desired
    - seed: int, the random seed to keep results constant but random
    - dataset_name: str,  a dataset for the model to be trained on
    - fold: str, a name to save the weights of the model
    - pretrained: str, the name of a preexisting model to be trained further

    example of use: model_trainig(100, False, 22, "dataset1", "model1" )"""

from unet_model_with_functions_of_blocks import build_unet
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from focal_loss import BinaryFocalLoss


#Dice metric can be a great metric to track accuracy of semantic segmentation.
def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    # if y_pred.sum() == 0 and y_pred.sum() == 0:
    #     return 1.0

    return 2*intersection / union

batch_size= 2



#If You need to resize images then add this to the flow_from_directory parameters 
#target_size=(150, 150), #Or whatever the size is for your network
def model_training(dest_path, epochs, augmented, seed, dataset_name, model_name, pretrained = "empty" ):
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
    image_generator = image_data_generator.flow_from_directory(dataset_name + "/train_images/", 
                                                               seed=seed, 
                                                               batch_size=batch_size,
                                                               color_mode = 'grayscale',
                                                               #target_size = (128,128),
                                                               class_mode=None)  #Very important to set this otherwise it returns multiple numpy arrays 
                                                                                #thinking class mode is binary.

    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_generator = mask_data_generator.flow_from_directory(dataset_name + "/train_masks/", 
                                                             seed=seed, 
                                                             batch_size=batch_size,
                                                             color_mode = 'grayscale',   #Read masks in grayscale
                                                             #target_size = (128,128),
                                                             class_mode=None)


    valid_img_generator = image_data_generator.flow_from_directory(dataset_name + "/val_images/", 
                                                                   seed=seed, 
                                                                   batch_size=batch_size,
                                                                   color_mode = 'grayscale', 
                                                                   #target_size = (128,128),
                                                                   class_mode=None) #Default batch size 32, if not specified here
    valid_mask_generator = mask_data_generator.flow_from_directory(dataset_name + "/val_masks/", 
                                                                   seed=seed, 
                                                                   batch_size=batch_size, 
                                                                   color_mode = 'grayscale',   #Read masks in grayscale
                                                                   #target_size = (128,128),
                                                                   class_mode=None)  #Default batch size 32, if not specified here


    train_generator = zip(image_generator, mask_generator)
    val_generator = zip(valid_img_generator, valid_mask_generator)

    x = image_generator.next()
    y = mask_generator.next()

    IMG_HEIGHT = x.shape[1]
    IMG_WIDTH  = x.shape[2]
    IMG_CHANNELS = x.shape[3]

    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    if pretrained != "empty":

        model = tf.keras.models.load_model(pretrained + ".hdf5", compile=False)
    else:

        model = build_unet(input_shape)

    model.compile(optimizer=Adam(lr = 1e-3), loss=BinaryFocalLoss(gamma=2), 
                  metrics=[dice_metric])

    model.summary()

    num_train_imgs = len(os.listdir(dataset_name + '/train_images/train/'))

    steps_per_epoch = num_train_imgs //batch_size

    checkpoint_path = os.path.join(dest_path,model_name + "_best.hdf5")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_best_only=True,
                                                 monitor = "loss",
                                                 verbose=1)

    history = model.fit_generator(train_generator, validation_data=val_generator, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_steps=steps_per_epoch, epochs=epochs,
                        callbacks = [cp_callback])

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig( os.path.join(dest_path,model_name + ".png"))
    model.save( os.path.join(dest_path, model_name + '.hdf5'))
    
    return model
