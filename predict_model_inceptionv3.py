"""
    Name:           train_model.py
    Created:        10/7/2017
    Description:    Fine-tune inception v3 for Planet Amazon.
"""
#==============================================
#                   Modules
#==============================================
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
import numpy as np
import pandas as pd
import time
import gzip
import pickle
from collections import Counter
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
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
from train_model_inceptionv3 import fbs, binary_crossentropy_weighted, preprocess_input


#==============================================
#                   Functions
#==============================================
def load_model(chosen_metrics=[fbs],
               inception_json="inceptionv3_mod.json",
               inception_h5="inceptionv3_fine_tuned_2.h5", verbose=1):
    """
    Load the inception v3 and trained weights from disk.
    """

    # load json and create model
    with open(inception_json, 'r') as iOF:
        loaded_model_json = iOF.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(inception_h5)
    if verbose >= 1: print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(lr=0.0005), loss=binary_crossentropy_weighted, metrics=[fbs])

    return loaded_model





def infer(model, X_test, y_test, n_inference=100, batch_size=32, verbose=1):
    """
    Infer with the inception v3.
    """

    # number of samples
    nb_test_samples = len(y_test)

    # add fake entries to get batch size multiple
    nb_add = batch_size - nb_test_samples%batch_size
    y_test_add = y_test[-nb_add:, ...]
    X_test_add = X_test[-nb_add:, ...]
    y_test_stacked = np.vstack([y_test, y_test_add])
    X_test_stacked = np.vstack([X_test, X_test_add])

    y_pred_list = []
    y_test_list = []
    for inference_pass in range(n_inference):

        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            rotation_range=180,
            fill_mode='reflect')

        test_generator = test_datagen.flow(
            X_test_stacked,
            y_test_stacked,
            batch_size=batch_size,
            shuffle=False)

        y_pred = model.predict_generator(
            test_generator,
            steps=(nb_test_samples + nb_add) // batch_size,
            verbose=1)

        # get predictions
        y_pred = y_pred[:-nb_add, ...]
        y_pred_list.append(y_pred)

        # append ground truth
        y_test = y_test_stacked[:-nb_add, ...]
        y_test_list.append(y_test)

    y_pred = np.array(y_pred_list)
    y_test = np.array(y_test_list)

    return y_pred, y_test



def main(verbose=1):
    """
    Main function.
    """

    if sys.argv[2] != 'test':

        ##### Validation
        fold_id = int(sys.argv[2])

        ### Load model
        if verbose >= 1: print("Loading model (fold %d)..."%fold_id)
        model_dir = "../data/planet_amazon/models/"
        model = load_model(inception_json=model_dir+"inceptionv3_mod_%d.json"%fold_id,
                           inception_h5=model_dir+"inceptionv3_fine_tuned_check_point_2_%d.h5"%fold_id,
                           verbose=verbose)

        ### Load images
        if verbose >= 1: print("Loading images into RAM (fold %d)..."%fold_id)
        image_dir="../data/planet_amazon/train-jpg/"
        target_size = (256,256)
        df_val = pd.read_csv("../data/planet_amazon/val%d.csv"%fold_id)
        X_val, y_val = [], []
        # for train and validation
        for image_id, y_lab in tqdm(list(zip(df_val.image_name, df_val.iloc[:,2:].values)), miniters=100):
            image_path = image_dir+str(image_id)+".jpg"
            if os.path.exists(image_path):
                try:
                    img = load_img(image_path, target_size=target_size)
                    arr = img_to_array(img)
                    X_val.append(arr)
                    y_val.append(y_lab)
                except OSError:
                    if verbose >= 2: print("OSError on image %s."%image_path)
            else:
                raise(ValueError("Image %s does not exist."%image_path))
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        if verbose >= 2:
            print(X_val.shape)
            print(y_val.shape)
            print(np.mean(y_val, axis=0))

        ### Infer
        if verbose >= 1: print("Inferring (fold %d)..."%fold_id)
        n_inference = 20
        y_pred, y_test = infer(model, X_val, y_val, n_inference=n_inference, batch_size=32, verbose=verbose)
        if verbose >= 2:
            print(y_pred.shape)
            print(y_test.shape)
            print(y_pred[0,0,0])
            print(y_test[0,0,0])
            print("Fbeta score: ", fbeta_score(np.mean(y_test, axis=0), np.mean(y_pred, axis=0).round(), 2, average='samples'))

        ### Save preds
        if verbose >= 1: print("Saving preds (fold %d)..."%fold_id)
        with open("../data/planet_amazon/inceptionv3_preds%d_%d.npy"%(fold_id,n_inference), "wb") as iOF:
            np.save(iOF, y_pred)
        with open("../data/planet_amazon/inceptionv3_trues%d_%d.npy"%(fold_id,n_inference), "wb") as iOF:
            np.save(iOF, y_test)

    else:

        ### Load images
        if verbose >= 1: print("Loading images into RAM...")
        image_dir="../data/planet_amazon/test-jpg/"
        target_size = (256,256)
        image_ids = [fname[:-4] for fname in os.listdir(image_dir) if fname.endswith(".jpg")]
        X_test, y_test = [], []
        # for train and validation
        for image_id in tqdm(image_ids, miniters=100):
            image_path = image_dir+str(image_id)+".jpg"
            if os.path.exists(image_path):
                try:
                    img = load_img(image_path, target_size=target_size)
                    arr = img_to_array(img)
                    X_test.append(arr)
                    y_lab = np.zeros((17,))
                    y_test.append(y_lab)
                except OSError:
                    if verbose >= 2: print("OSError on image %s."%image_path)
            else:
                raise(ValueError("Image %s does not exist."%image_path))
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        if verbose >= 2:
            print(X_test.shape)
            print(y_test.shape)
            print(np.mean(y_test, axis=0))

        ##### Test
        y_pred_folds = []
        offset = 0
        for fold_id in range(offset,offset+5):
            ### Load model
            if verbose >= 1: print("Loading model (fold %d)..."%fold_id)
            model_dir = "../data/planet_amazon/models/"
            model = load_model(inception_json=model_dir+"inceptionv3_mod_%d.json"%fold_id,
                               inception_h5=model_dir+"inceptionv3_fine_tuned_check_point_2_%d.h5"%fold_id,
                               verbose=verbose)

            ### Infer
            if verbose >= 1: print("Inferring (fold %d)..."%fold_id)
            n_inference = 20
            y_pred, _ = infer(model, X_test, y_test, n_inference=n_inference//5, batch_size=32, verbose=verbose)
            y_pred_folds.append(y_pred)
            if verbose >= 2:
                print(y_pred.shape)
                print(y_pred[0,0,0])

        y_pred_folds = np.vstack(y_pred_folds)
        if verbose >= 2:
            print(y_pred_folds.shape)
            print(y_pred_folds[0,0,0])

        ### Save preds
        if verbose >= 1: print("Saving preds...")
        with open("../data/planet_amazon/inceptionv3_predstest_%d.npy"%n_inference, "wb") as iOF:
            np.save(iOF, y_pred_folds)
        with open("../data/planet_amazon/inceptionv3_idstest_%d.txt"%n_inference, "w") as iOF:
            iOF.writelines([image_id+"\n" for image_id in image_ids])





#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    main(2)
