import os
import pickle
import argparse
import pandas as pd
import numpy as np
import re
from os import walk
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
import random
import shutil
import sys
import csv
import warnings
from sklearn.metrics import f1_score

import logging
import time
import math
from pathlib import Path
from keras import optimizers
from keras.layers import GlobalAveragePooling2D
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet import ResNet101
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.layers import Dense
from keras.models import Model, load_model
from keras.preprocessing import image
from tqdm import tqdm
import tensorflow as tf
pd.options.mode.chained_assignment = None
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




def train_model_get_threshold(train_dir, val_dir, cnn_model, batch_size, learning_rate, epochs, patience):
    
    """
    Arguments:
    train_dir == directory that contains the (radiology)train dataset images and cui.txt(concepts for the images)
    val_dir == directory that contains the (radiology)validation dataset images and cui.txt(concepts for the images)
    cnn_model == baseline CNN models used, "resnet101" or "chexnet", by default --> it is set to 'chexnet', you can change it
    batch_size == 128 by default, 
    learning_rate == 0.0001 by default, 
    epochs == 30 by default, 
    patience == 5 by default,

    ###batch_size, epochs, learning_rate and patience are all changeable for hyperparameter fine-tuning###
    
    ##########
    Run this function, For Example via Command line:
    python dir/train_model_get_threshold.py train_dir val_dir
    ###########
    
    Outputs:
    This function outputs:
    Model_path --> train_dir/model/model_checkpoint.hdf5
    concepts used for training --> train_dir/concept_list.pickle
    best_decision threshold needed for F1 Score --> train_dir/threshold.pickle

    ###train_dir --> where your train dataset directory is located.

    This function perform the following tasks:
    -- Create df for both train and val data
    -- Preprocess the data --> Generators for the datasets and Multilabelbinarizer
    -- Model training
    -- Plot the model loss history -- to evaluate model training
    -- evaluate results by F1 Score based on different thresholds, -- and choose the best threshold with highest F1 Score
    -- Save the concepts used for model training, and the best decision threshold -- both are needed for prediction on test data
    """
    
    def create_df(data_dir):
        
        """
        This function creates dataframe with images names and CUIS columns
        """
        
        ##get the number of images in the images dir
        data_img = os.listdir(os.path.join(data_dir, 'images'))
        ##sort the list
        data_img.sort()
        ##df
        df = pd.DataFrame(data_img, columns = ['image_name'])
        
        ##read in the cuis
        df_cuis = pd.read_fwf(os.path.join(data_dir, 'cuis.txt'), header = None)
        cols = df_cuis.columns.tolist()[1:]
        df_cuis['concepts'] = df_cuis[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        
        ##new df
        df_cuis = df_cuis.loc[:, [0, 'concepts']].copy()
        df_cuis['concepts'] = df_cuis['concepts'].apply(lambda x: re.sub(r'nan', '', x))
        df_cuis = df_cuis[df_cuis['concepts'] != '']
        df_cuis['concepts'] = df_cuis['concepts'].str.split()
        
        ###Lets Concatenate image_names, concepts to one df
        df = pd.concat([df, df_cuis], axis=1, join='inner')
        
        df.drop([0], axis = 1, inplace = True)
        df = df[df['concepts'].map(lambda d: len(d)) > 0] ##exclude rows with empty concept list
        df.reset_index(inplace = True)
        df.drop(['index'], axis = 1, inplace = True)
        ##create a new column "image_path"
        df['image_path'] = df['image_name'].apply(lambda x:'images/' + x)
        
        return df
    
    df_train = create_df(train_dir)
    df_val = create_df(val_dir)
    
    ##concatenate val and train
    df_val_trn = pd.concat([df_train, df_val], ignore_index=True)
    
    
    
    ###DATA PROCESSING
    ##augment the data by horizontal flip
    ## normalize the data scaling it down to btw 0 and 1 
    ##CREDITS: https://github.com/AntonisFaros/Image_Classification_IMAGECLEF2019/blob/master/ImageClef2019-Faros%20Antonios.ipynb
    train_image_generator = ImageDataGenerator(rescale=1./255,
                                               shear_range = 0.2,
                                               zoom_range = 0.2,
                                               horizontal_flip = True) # Generator for our training data
    val_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
    dev_image_generator = ImageDataGenerator(rescale = 1./255)#Generator for the valiation evaluation
    
    
    ##train and validation datasets
    df_train_concepts = df_val_trn.loc[0:df_train.shape[0]-1]
    df_val_concepts = df_val_trn.loc[df_train.shape[0]:]
    df_val_concepts.reset_index(inplace = True)
    df_val_concepts.drop(['index'], axis = 1, inplace = True)
    
    ##get the index as a column
    ##this will be needed to apply multilabel encoding
    df_train_concepts['index'] = df_train_concepts.index
    df_val_concepts['index'] = df_val_concepts.index
    
    ##image heights & width, batch_size
    img_height = 224
    img_width = 224
    batch_size = batch_size
    
    ##Generators for train and validation datasets
    def img_generator(df_train_concepts, train_dir, df_val_concepts, val_dir, batch_size, img_height, img_width):
        
        ##train_generator
        train_data_gen = train_image_generator.flow_from_dataframe(dataframe= df_train_concepts,
                                                                   directory = train_dir,
                                                                   x_col="image_path",
                                                                   y_col="index",
                                                                   batch_size=batch_size,
                                                                   shuffle=True,
                                                                   target_size=(img_height, img_width),
                                                                   class_mode = 'raw')
        
        ##validation_generator -- still part of training that's why shuffling == True
        val_data_gen = val_image_generator.flow_from_dataframe(dataframe = df_val_concepts,
                                                               directory=val_dir,
                                                               x_col = "image_path",
                                                               y_col = "index",
                                                               shuffle = True,
                                                               batch_size=batch_size,
                                                               target_size=(img_height, img_width),
                                                               class_mode='raw')
        
        ##validation dataset that will be used to tune the decision threshold to get the optimal F1 Score
        dev_data_gen = dev_image_generator.flow_from_directory(directory=val_dir,
                                                               shuffle = False,
                                                               batch_size=batch_size,
                                                               target_size=(img_height, img_width), 
                                                               class_mode = None)
        
        return train_data_gen, val_data_gen, dev_data_gen
    
    train_data_gen, val_data_gen, dev_data_gen = img_generator(df_train_concepts, train_dir, df_val_concepts, 
                                                               val_dir, batch_size, img_height, img_width)
    
    
    
    ##Multilabel Binarizer
    ##-- generate the full set of labels for each data sample in binarize format (0,1)
    def mlbinarize(df):
        
        ##Transform the labels i.e. concepts into a Binary formats
        mlb = MultiLabelBinarizer()
        mlb.fit(df['concepts'])
        num_of_labels = mlb.classes_
        ##validation + training data labels
        print('Total number of labels:', len(num_of_labels))
        
        return mlb, num_of_labels
    
    mlb, num_of_labels = mlbinarize(df_val_trn)
    
    ## this helper function binarize the labels of each images 
    ##if a concept is present in the image, the value = 1
    ##And if the concepts is not present, the value is 0
    def multilabel_flow_from_dataframe(data_generator, mlb, df):
        while True:
            for x, y in data_generator:
                indices = y.astype(np.int).tolist()
                y_multilabel = mlb.transform(df.iloc[indices]['concepts'].values.tolist())
                yield x, y_multilabel
    
    
    ##multilabel_generator -> x == images, y == total num of labels for each image; per Batch
    multilabel_generator_train = multilabel_flow_from_dataframe(train_data_gen, mlb, df_train_concepts)
    multilabel_generator_val = multilabel_flow_from_dataframe(val_data_gen, mlb, df_val_concepts)
    
    ####MODEL TRAINING
    
    ##check if the model folder already exist in the train_dir...if not create a new one
    if not os.path.exists(os.path.join(train_dir, 'model')):
        os.mkdir(os.path.join(train_dir, 'model'))
    
    model_dir = os.path.join(train_dir, 'model')
    
    ##CREDITS: https://github.com/nlpaueb/bioCaption/blob/master/bioCaption/models/tagModels/chexnet.py -- bestmodel ImageCLEF 2019
    def chexnet_model(num_tags):
        my_init = glorot_uniform(seed=42)
        base_model = DenseNet121(weights='imagenet', include_top=True)
        x = base_model.get_layer("avg_pool").output
        concept_outputs = Dense(num_tags, activation="sigmoid", 
                                name="concept_outputs", kernel_initializer=my_init)(x)
        model = Model(inputs=base_model.input, outputs=concept_outputs)
        ###model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
        opt = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["binary_accuracy"])
        return model
    
    ###RESNET 101 --
    def resnet101_model(num_tags):
        my_init = glorot_uniform(seed=42)
        base_model = ResNet101(include_top=False, weights='imagenet')
        x = base_model.output
        # Adding a Global Average Pooling layer
        x = GlobalAveragePooling2D()(x)
        concept_outputs = Dense(num_tags, activation="sigmoid",
                                name="concept_outputs", kernel_initializer=my_init)(x)
        model = Model(inputs=base_model.input, outputs=concept_outputs)
        ###model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
        opt = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["binary_accuracy"])
        return model
    
    ##load model --- uu could add other state-of-the-art-cnn-models
    if cnn_model == 'chexnet':
        model_name = chexnet_model(len(num_of_labels))
    elif cnn_model == 'resnet101':
        model_name = resnet101_model(len(num_of_labels))
    
    
    epochs = epochs
    
    ##Early Stopping -- Stop training when the validation loss is no more reducing after waiting for 3 epochs (patience = 3).
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience,
                                   mode="auto", restore_best_weights=True)
    
    # save best model
    checkpoint = ModelCheckpoint(os.path.join(model_dir, "model_checkpoint.hdf5"), 
                                 monitor="val_loss", save_best_only=True, mode="auto")
    
    ##Reduce Learning rate -- Reduce learning rate when the validation loss stop reducing by a factor of 0.1
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, verbose=1, mode="min")
    
    # train the model
    model_history = model_name.fit(multilabel_generator_train, 
                                  steps_per_epoch= math.ceil(df_train_concepts.shape[0] / batch_size), 
                                  epochs=epochs, callbacks=[early_stopping, checkpoint, reduce_lr], 
                                  validation_data= multilabel_generator_val, 
                                  validation_steps= math.ceil(df_val_concepts.shape[0] / batch_size),
                                   workers = 4, use_multiprocessing=True)
    
    ##plot model loss history
    def plot_history(history):
        plt.figure(figsize=(8,4))
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('LSTM - train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper right')

    plot_history(model_history)
    plt.show(block=True)
    
    ###MODEL PREDICTION AND EVALUATION TO GET THE BEST THRESHOLD FOR THE HIGHEST F1 SCORE####
    
    def get_predictions(model_path, generator_data):
        cnn_model = load_model(model_path)
        predictions = cnn_model.predict(generator_data)
        return predictions
    
    ###get the validation dictionary and its predicted counterpact
    def results_groundtruth_data(predictions, data_dir, decision_threshold):
        
        #validation images
        val_images = os.listdir(data_dir + 'images')
        df_val_images = pd.DataFrame(val_images, columns = ['image_name'])
        
        ##get the predicted results/concepts with their corresponding validation images
        val_results = {}
        for i in range(len(predictions)):
            concepts_pred = []
            for j in range(len(mlb.classes_)):
                if predictions[i, j] >= decision_threshold:
                    concepts_pred.append(mlb.classes_[j])
            val_results[val_images[i]] = ";".join(concepts_pred)
            
        ##read in the cuis (concepts) for the validation set
        df_cuis = pd.read_fwf(data_dir + 'cuis.txt', header = None)
        cols = df_cuis.columns.tolist()[1:]
        df_cuis['concepts'] = df_cuis[cols].apply(lambda row: ';'.join(row.values.astype(str)), axis=1)
        
        ##new df_cuis
        df_cuis = df_cuis.loc[:, [0, 'concepts']].copy()
        df_cuis['concepts'] = df_cuis['concepts'].apply(lambda x: re.sub(r'nan;|;nan|nan', '', x))
        
        ###Lets Concatenate validate_images and cuis df to one df
        df_gd_val = pd.concat([df_val_images, df_cuis], axis=1, join='inner')
        df_gd_val.drop([0], axis = 1, inplace = True)
        
        ##get the groundtruth concepts with their corresponding images
        val_groundtruth = {}
        for x in range(df_gd_val.shape[0]):
            val_groundtruth[df_gd_val['image_name'][x]] = df_gd_val['concepts'][x]
            
        return val_groundtruth, val_results
    
    model_path = os.path.join(model_dir, "model_checkpoint.hdf5")
    dev_predictions = get_predictions(model_path, dev_data_gen)
    
    
    ##-- ImageCLEF 2019 F1 Evaluation Function (Remodified)
    def calculate_f1(candidate_pairs, gt_pairs):
        
        # Hide warnings
        warnings.filterwarnings('ignore')
        
        # Concept stats
        min_concepts = sys.maxsize
        max_concepts = 0
        total_concepts = 0
        concepts_distrib = {}
        
        # Read files
        #print('Input parameters\n********************************')
        
        # Define max score and current score
        max_score = len(gt_pairs)
        current_score = 0
        
        # Check there are the same number of pairs between candidate and ground truth
        if len(candidate_pairs) != len(gt_pairs):
            print('ERROR : Candidate does not contain the same number of entries as the ground truth!')
            exit(1)
        
        # Evaluate each candidate concept list against the ground truth
        #print('Processing concept sets...\n********************************')
        
        i = 0
        for image_key in candidate_pairs:
            
            # Get candidate and GT concepts
            candidate_concepts = candidate_pairs[image_key].upper()
            gt_concepts = gt_pairs[image_key].upper()
            
            # Split concept string into concept array
            # Manage empty concept lists
            if gt_concepts.strip() == '':
                gt_concepts = []
            else:
                gt_concepts = gt_concepts.split(';')
                
            if candidate_concepts.strip() == '':
                candidate_concepts = []
            else:
                candidate_concepts = candidate_concepts.split(';')
                
            # Manage empty GT concepts (ignore in evaluation)
            if len(gt_concepts) == 0:
                max_score -= 1
            # Normal evaluation
            else:
                # Concepts stats
                total_concepts += len(gt_concepts)
                
                # Global set of concepts
                all_concepts = sorted(list(set(gt_concepts + candidate_concepts)))
                
                # Calculate F1 score for the current concepts
                y_true = [int(concept in gt_concepts) for concept in all_concepts]
                y_pred = [int(concept in candidate_concepts) for concept in all_concepts]
                
                f1score = f1_score(y_true, y_pred, average='binary')
                
                # Increase calculated score
                current_score += f1score
                
            # Concepts stats
            nb_concepts = str(len(gt_concepts))
            if nb_concepts not in concepts_distrib:
                concepts_distrib[nb_concepts] = 1
            else:
                concepts_distrib[nb_concepts] += 1
            if len(gt_concepts) > max_concepts:
                max_concepts = len(gt_concepts)
            if len(gt_concepts) < min_concepts:
                min_concepts = len(gt_concepts)
        
        mean_f1 = current_score / max_score
        
        return mean_f1
    
    #lets calculate the f1 scores on different "Decision Threshold" values and see which one has the highest f1 score
    ##based on previous papers on imageclef 2019, let calculate f1 for the threshold btw 0.1 and 0.21
    threshold_dict = {}
    for x in np.arange(0.1, 0.21, 0.01):
        val_gt, val_candidate = results_groundtruth_data(dev_predictions, val_dir, decision_threshold = x)
        threshold_f1score = calculate_f1(val_candidate, val_gt)
        threshold_dict[x] = threshold_f1score
        
    best_threshold = max(threshold_dict, key=lambda k: threshold_dict[k])
    
    print('\n The decision threshold for calculating F1 score on the test data will be:', best_threshold)
    print('\n Highest F1 Score for the validation dataset:', threshold_dict[best_threshold])
    
    ##Save the number of classes and the threshold score -- this is needed for predicting concepts for 'test images'
    def save_to_pickle(file_dir, file, name):
        with open(file_dir + name + '.pickle', 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ##save the classes/concepts used to train the model -- to train_dir
    save_to_pickle(train_dir, num_of_labels, 'concept_list')
    print('\n Saved the Concepts used to evaluate the model in: \n',train_dir +'concept_list.pkl, will be needed for prediction on test/unseen data')
    
    ##save decision threshold value
    save_to_pickle(train_dir, best_threshold, 'threshold')
    print('\n Saved the best decision threshold in: \n',train_dir +'threshold.pkl, will ONLY be needed for prediction on test data')


###Arguments to pass to this Script
def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments")
    ###positional arguments
    ##file directories
    parser.add_argument('train_dir', type=str, help='dir that contains train images and cui.txt->(Concepts for the images)')
    parser.add_argument('val_dir', type=str, help='dir that contains validation images and cui.txt->(Concepts for the images)')
    ##model hyperparameters
    parser.add_argument('cnn_model', type=str, default = 'chexnet', nargs='?', const = 'chexnet', help='Choose one of the models, type in one of them: "chexnet" or "resnet101"')
    parser.add_argument('batch_size', type=int, default = 128, nargs='?', const = 16, help='Batch size for the model inputs')
    parser.add_argument('learning_rate', type=float, default = 0.0001, nargs='?', const = 0.001, help='learning rate ')
    parser.add_argument('epochs', type=int, default = 30, nargs='?', const = 10, help='Number of epochs')
    parser.add_argument('patience', type=int, default = 5, nargs='?', const = 5, help='Stop training or reduce the learning rate when the val loss is no more reducing after waiting for the amount of "patience"')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_model_get_threshold(args.train_dir, args.val_dir, args.cnn_model, args.batch_size, 
                              args.learning_rate, args.epochs, args.patience)