"""
    Name:           train_model.py
    Created:        10/7/2017
    Description:    Fine-tune vgg 16 for Planet Amazon.
"""
#==============================================
#                   Modules
#==============================================
import os
import sys
gpu_id = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
import numpy as np
import pandas as pd
import time
import gzip
import pickle
from collections import Counter
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model, model_from_json
from keras.layers import Dense, Flatten, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.imagenet_utils import decode_predictions
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score
from tqdm import tqdm
#==============================================
#                   Files
#==============================================
_EPSILON = K.epsilon()


#==============================================
#                   Functions
#==============================================
def instantiate(n_classes, n_dense=1024, vgg_json="vgg16_mod.json", target_size=(256,256,3), verbose=1):
    """
    Instantiate the vgg 16.
    """

    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=target_size)

    # add a global spatial average pooling layer
    x = base_model.output
    x = Flatten()(x)
    # let's add fully-connected layers
    x = Dense(n_dense, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(n_dense, activation='relu')(x)
    x = Dropout(0.2)(x)
    # and a final logistic layer
    predictions = Dense(n_classes, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=Adam(lr=0.0005), loss=binary_crossentropy_weighted, metrics=[fbs])

    # serialize model to json
    model_json = model.to_json()
    with open(vgg_json, "w") as iOF:
        iOF.write(model_json)

    return base_model, model




def finetune(base_model, model, X_train, y_train, X_val, y_val,
             epochs_1=1000, patience_1=2,
             patience_lr=1, batch_size=32,
             nb_train_samples=41000, nb_validation_samples=7611,
             img_width=299, img_height=299, class_imbalance=False,
             vgg_h5_1="vgg16_fine_tuned_1.h5",
             vgg_h5_check_point_1="vgg16_fine_tuned_check_point_1.h5",
             layer_names_file="vgg16_mod_layer_names.txt", verbose=1):
    """
    Finetune the vgg 16.
    """

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    with open(layer_names_file, "w") as iOF:
        for ix, layer in enumerate(base_model.layers):
            iOF.write("%d, %s\n"%(ix, layer.name))
            if verbose >= 4: print(ix, layer.name)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=180,
        fill_mode='reflect')

    # this is the augmentation configuration we will use for testing:
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=180,
        fill_mode='reflect')

    # define train & val data generators
    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True)

    validation_generator = test_datagen.flow(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=True)

    # get class weights
    if class_imbalance:
        class_weight = get_class_weights(np.sum(y_train, axis=0), smooth_factor=0.1)
    else:
        class_weight = None

    # train the model on the new data for a few epochs on the batches generated by datagen.flow().
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs_1,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience_1),
                   ModelCheckpoint(filepath=vgg_h5_check_point_1, save_best_only=True),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience_lr)],
        class_weight=class_weight)

    # save weights just in case
    model.save_weights(vgg_h5_1)




def finetune_from_saved(vgg_h5_load_from, vgg_h5_save_to,
             vgg_json, X_train, y_train, X_val, y_val, nb_freeze=11,
             epochs=5000, patience=2, patience_lr=1, batch_size=32,
             nb_train_samples=85639, nb_validation_samples=10694,
             img_width=299, img_height=299, class_imbalance=False, optimizer_lr=0.0002,
             vgg_h5_check_point="vgg16_fine_tuned_check_point_2.h5", verbose=1):
    """
    Finetune the vgg 16 from already fine-tuned one.
    """

    # load json and create model
    with open(vgg_json, 'r') as iOF:
        loaded_model_json = iOF.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(vgg_h5_load_from)
    if verbose >= 1: print("Loaded model from disk")

    # we freeze the first nb_freeze layers and unfreeze the rest:
    for layer in loaded_model.layers[:nb_freeze]:
        layer.trainable = False
    for layer in loaded_model.layers[nb_freeze:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    loaded_model.compile(optimizer=Adam(lr=optimizer_lr), loss=binary_crossentropy_weighted, metrics=[fbs])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=180,
        fill_mode='reflect')

    # this is the augmentation configuration we will use for testing:
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=180,
        fill_mode='reflect')

    # define train & val data generators
    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True)

    validation_generator = test_datagen.flow(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=True)

    # get class weights
    if class_imbalance:
        class_weight = get_class_weights(np.sum(y_train, axis=0), smooth_factor=0.1)
    else:
        class_weight = None

    # train the model on the new data for a few epochs on the batches generated by datagen.flow().
    loaded_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience),
                   ModelCheckpoint(filepath=vgg_h5_check_point, save_best_only=True),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience_lr)],
        class_weight=class_weight)

    # save weights
    loaded_model.save_weights(vgg_h5_save_to)





def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """

    if smooth_factor > 0:
        p = y.max() * smooth_factor
        y = y + p

    majority = float(y.max())

    return {clss: (majority / cnt) for clss, cnt in enumerate(y)}



def f2_score(y_true, y_pred):
    # fbs throws a confusing error if inputs are not numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbs(y_true, y_pred, beta=2, average='samples')



def fbs(y_true, y_pred, threshold_shift=0., beta=2):

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())



def binary_crossentropy_weighted(y_true, y_pred, one_weight=4.):
    y_weight = K.clip(y_true * one_weight, 1., one_weight)
    out = K.binary_crossentropy(y_pred, y_true) * y_weight
    return K.mean(out, axis=-1)




def train_for_a_fold(df_train, df_val, fold_id, target_size=(256,256),
                     model_dir="../data/planet_amazon/models/",
                     image_dir="../data/planet_amazon/train-jpg/",
                     verbose=1):
    """
    Train an VGG16 for a fold.
    """

    if verbose >= 1: print("Training for fold %d..."%fold_id)

    ### Prepare weights
    labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',\
              'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation',\
              'habitation', 'haze', 'partly_cloudy', 'primary', 'road',\
              'selective_logging', 'slash_burn', 'water']
    n_labels = np.array([12315, 339, 862, 332, 98, 28431,
                         9350, 100, 4477, 3660, 2697, 7261,
                         37513, 8071, 340, 209, 7411])
    label_weights = np.array([cw for cid, cw in sorted(get_class_weights(n_labels).items())])
    label_counts = np.ceil( 10. * label_weights / label_weights.max() ).astype(int)

    ### Load images
    if verbose >= 1: print("\tLoading images into RAM (fold %d)..."%fold_id)
    X_train, y_train = [], []
    X_val, y_val = [], []
    # for train and validation
    for df, X, y, n_max_img in [(df_train, X_train, y_train, 10), (df_val, X_val, y_val, 1)]:
        for image_id, y_lab in tqdm(list(zip(df.image_name, df.iloc[:,2:].values)), miniters=100):
            image_path = image_dir+str(image_id)+".jpg"
            if os.path.exists(image_path):
                try:
                    img = load_img(image_path, target_size=target_size)
                    arr = img_to_array(img)
                    for _ in range(min((int(np.max(label_counts * y_lab)), n_max_img))):
                        X.append(arr)
                        y.append(y_lab)
                except OSError:
                    if verbose >= 2: print("OSError on image %s."%image_path)
            else:
                raise(ValueError("Image %s does not exist."%image_path))
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    if verbose >= 2:
        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)
        print(np.mean(y_train, axis=0))
        print(np.mean(y_val, axis=0))

    ### Create model
    if verbose >= 1: print("\tInstantiating VGG16 (fold %d)..."%fold_id)
    n_classes = y_train.shape[1]
    base_model, model = instantiate(n_classes, n_dense=1024, vgg_json=model_dir+"vgg16_mod_%d.json"%fold_id, verbose=verbose)

    ### Train model
    if verbose >= 1: print("\tFine-tuning VGG16 first pass (fold %d)..."%fold_id)
    finetune(base_model, model, X_train, y_train, X_val, y_val, batch_size=32, epochs_1=5,
             nb_train_samples=len(y_train), nb_validation_samples=len(y_val),
             patience_1=2, patience_lr=1, class_imbalance=True,
             vgg_h5_1=model_dir+"vgg16_fine_tuned_1_%d.h5"%fold_id,
             vgg_h5_check_point_1=model_dir+"vgg16_fine_tuned_check_point_1_%d.h5"%fold_id,
             layer_names_file=model_dir+"vgg16_mod_layer_names.txt",
             verbose=verbose)
    del(base_model)
    del(model)
    K.clear_session()
    if verbose >= 1: print("\tFine-tuning VGG16 second pass (fold %d)..."%fold_id)
    finetune_from_saved(model_dir+"vgg16_fine_tuned_check_point_1_%d.h5"%fold_id,
                        model_dir+"vgg16_fine_tuned_2_%d.h5"%fold_id,
                        model_dir+"vgg16_mod_%d.json"%fold_id,
                        X_train, y_train, X_val, y_val, batch_size=32, epochs=10, optimizer_lr=0.0002,
                        nb_freeze=11, patience=2, patience_lr=1, class_imbalance=True,
                        nb_train_samples=len(y_train), nb_validation_samples=len(y_val),
                        vgg_h5_check_point=model_dir+"vgg16_fine_tuned_check_point_2_%d.h5"%fold_id,
                        verbose=verbose)
    K.clear_session()
    if verbose >= 1: print("\tFine-tuning VGG16 third pass (fold %d)..."%fold_id)
    finetune_from_saved(model_dir+"vgg16_fine_tuned_check_point_2_%d.h5"%fold_id,
                        model_dir+"vgg16_fine_tuned_3_%d.h5"%fold_id,
                        model_dir+"vgg16_mod_%d.json"%fold_id,
                        X_train, y_train, X_val, y_val, batch_size=32, optimizer_lr=0.00002,
                        nb_freeze=0, patience=10, patience_lr=3, class_imbalance=True,
                        nb_train_samples=len(y_train), nb_validation_samples=len(y_val),
                        vgg_h5_check_point=model_dir+"vgg16_fine_tuned_check_point_3_%d.h5"%fold_id,
                        verbose=verbose)
    K.clear_session()




#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    fold_id = int(sys.argv[2])
    df_train = pd.read_csv("../data/planet_amazon/train%d.csv"%fold_id)
    df_val = pd.read_csv("../data/planet_amazon/val%d.csv"%fold_id)
    train_for_a_fold(df_train, df_val, fold_id, verbose=2)