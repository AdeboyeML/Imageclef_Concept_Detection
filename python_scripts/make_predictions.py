import os
import pickle
import argparse
import pandas as pd
import numpy as np
import re
from os import walk
from collections import Counter
from PIL import Image
import shutil
import sys
import csv
import warnings
from sklearn.metrics import f1_score

import logging
import time
import math
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import glorot_uniform
from keras.layers import Dense
from keras.models import Model, load_model
from keras.preprocessing import image
from tqdm import tqdm
import tensorflow as tf
pd.options.mode.chained_assignment = None
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




def make_predictions(test_dir, train_dir, batch_size):
    
    """
    Arguments:
    test_dir == directory that contains the (radiology) test dataset images and cui.txt(concepts for the images)
    train_dir == directory that contains the input files and models needed -- model_checkpoint, concept_list, threshold files
    batch_size == 128 -- by default, you can change it
    
    ##############
    Run this Function, For Example via Command line:
    python dir/make_predictions.py test_dir train_dir
    #############
    
    #####################
    ##NOTE: We need the train_dir because it contains the input files needed.
    
    This function outputs:
    -- groundtruth_predicted.csv --> test_dir/groundtruth_predicted.csv
    
    ##test_dir --> directory that contains the test dataset
    
    This function perform the following tasks:
    -- predicts concepts on test data
    -- Save the predicted and groundtruth results of the test data images to a csv file.
    #########################
    """
    
    ##input files
    model_path = os.path.join(train_dir, 'model/model_checkpoint.hdf5')
    concept_file = os.path.join(train_dir, 'concept_list.pickle')
    threshold_file = os.path.join(train_dir, 'threshold.pickle')
    
    ###DATA PROCESSING
    ## normalize the data scaling it down to btw 0 and 1 
    ##CREDITS: https://github.com/AntonisFaros/Image_Classification_IMAGECLEF2019/blob/master/ImageClef2019-Faros%20Antonios.ipynb
    test_image_generator = ImageDataGenerator(rescale = 1./255)#Generator for the valiation evaluation
    
    ##image heights & width
    img_height = 224
    img_width = 224
    batch_size = batch_size
    
    test_data_gen = test_image_generator.flow_from_directory(directory=test_dir, 
                                                             shuffle = False,
                                                             batch_size=batch_size,
                                                             target_size=(img_height, img_width),
                                                             class_mode = None)
    
    ###MODEL PREDICTION####
    
    def get_predictions(model_path, generator_data):
        cnn_model = load_model(model_path)
        predictions = cnn_model.predict(generator_data)
        return predictions
    
    ###get the validation dictionary and its predicted counterpact
    def results_groundtruth_data(predictions, data_dir, concept_lists, decision_threshold):
        
        #validation images
        val_images = os.listdir(data_dir + 'images')
        df_val_images = pd.DataFrame(val_images, columns = ['image_name'])
        
        ##get the predicted results/concepts with their corresponding validation images
        val_results = {}
        for i in range(len(predictions)):
            concepts_pred = []
            for j in range(len(concept_lists)):
                if predictions[i, j] >= decision_threshold:
                    concepts_pred.append(concept_lists[j])
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
    
    
    ##Load data (deserialize)
    def load_pickle(filename):
        with open(filename, 'rb') as handle:
            result = pickle.load(handle)
        return result
    
    concept_lists = load_pickle(concept_file)
    decision_threshold = load_pickle(threshold_file)
    
    test_predictions = get_predictions(model_path, test_data_gen)
    test_gt, test_candidate = results_groundtruth_data(test_predictions, test_dir, concept_lists, decision_threshold)
    test_f1score = calculate_f1(test_candidate, test_gt)
    
    print('\n The F1 Score for the test data:', test_f1score, '\n')
    
    ###save the predicted and the ground truth concepts as .CSV files
    ##First, create dataframe for both predicted and groundtruth data
    df_gt = pd.DataFrame(test_gt.items(), columns = ['image_name', 'Ground_truth Concepts'])
    df_pred = pd.DataFrame(test_candidate.items(), columns = ['image_name', 'Predicted Concepts'])
    #merge them
    df_gt_pred = pd.merge(df_gt, df_pred, on='image_name')
    
    ##save to csv
    df_gt_pred.to_csv(test_dir + 'groundtruth_predicted.csv', index = False)
    print('\n Saved the Ground truth and Predicted test concepts for corresponding images to -->\n', 
          os.path.join(test_dir, 'groundtruth_predicted.csv'))
    
    
###Arguments to pass to this Script
def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments")
    ###positional arguments
    ##file directories
    parser.add_argument('test_dir', type=str, help='dir that contains test images and cui.txt->(Concepts for the images)')
    parser.add_argument('train_dir', type=str, help='dir that contains train images and cui.txt->(Concepts for the images)')
    parser.add_argument('batch_size', type=int, default = 128, nargs='?', const = 16, help='Batch size for the model inputs')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    make_predictions(args.test_dir, args.train_dir, args.batch_size)