import os
import pickle
import argparse
import pandas as pd
import numpy as np
import re
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import random
import seaborn as sns
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

from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.initializers import glorot_uniform
from keras.layers import Dense
from keras.models import Model, load_model
from keras.preprocessing import image
from tqdm import tqdm
import tensorflow as tf
pd.options.mode.chained_assignment = None


def knn_model(train_dir, val_dir, test_dir):
    
    
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
        df['image_path'] = df['image_name'].apply(lambda x:data_dir + '/images/' + x)
        
        return df
    
    df_train = create_df(train_dir)
    df_val = create_df(val_dir)
    df_test = create_df(test_dir)
    
    ##train_dict with image_name as key, and concepts as value
    train_dict = {key:value for (key,value) in zip(df_train['image_name'], df_train['concepts'])}
    
    ##convert concept_list to str
    val_concepts = df_val['concepts'].apply(lambda row: ';'.join(row))
    tst_concepts = df_test['concepts'].apply(lambda row: ';'.join(row))
    
    ##train_dict with image_name as key, and concepts as value
    val_dict = {key:value for (key,value) in zip(df_val['image_name'], val_concepts)}
    test_dict = {key:value for (key,value) in zip(df_test['image_name'], tst_concepts)}
    
    #get img vectors
    def compute_image_embedding(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_vec= image.img_to_array(img)
        img_vec = np.expand_dims(img_vec, axis=0)
        img_vec = preprocess_input(img_vec)
        return img_vec
    
    
    def knn(df, df2):
        
        """
        Returns image_similarities for the validation data based on train vecs, a list of training image names, 
        and normalized image embeddings/dense vector encodings of the training images
        """
        
        base_model = DenseNet121(weights='imagenet', include_top=True)
        vector_extraction_model = \
        Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        print("Calculating visual embeddings from train images")
        train_images_vec = {}
        print("Extracting image embeddings (Dense vector encoding) for all train images...")
        for i, train_image in tqdm(enumerate(df['image_name'])):
            image_path = df['image_path'][i]
            x = compute_image_embedding(image_path)
            vec = vector_extraction_model.predict(x).transpose().flatten()
            train_images_vec[train_image] = vec
            #print("Got embeddings for train images.")
        
        # save image_name and raw image vectors seperately but aligned
        image_name = [i for i in train_images_vec]
        image_vectors = np.array([train_images_vec[i] for i in train_images_vec])
        # normalize image vectors to avoid normalized cosine and use dot
        image_vectors = image_vectors / np.array([np.sum(image_vectors, 1)] * image_vectors.shape[1]).transpose()
        
        # measure the similarity of each val image embedding with all train image embeddings
        images_similarity = {}
        for i, val_image in tqdm(enumerate(df2['image_name'])):
            image_path = df2['image_path'][i]
            x = compute_image_embedding(image_path)
            vec = vector_extraction_model.predict(x).transpose().flatten()
            vec = vec / np.sum(vec)
            # clone to do efficient mat mul dot
            test_mat = np.array([vec] * image_vectors.shape[0])
            sims = np.sum(test_mat * image_vectors, 1)
            # save the similarities array for every test image
            ##the no of cosine similarities per val image correspond to the nos of training images
            images_similarity[val_image] = sims
        
        return images_similarity, image_name, image_vectors
    
    img_sims, img_names, img_vects = knn(df_train, df_val)
    
    ##-- ImageCLEF 2019 F1 Evaluation Function (Remodified)
    def calculate_f1(candidate_pairs, gt_pairs):
        
        # Hide warnings
        warnings.filterwarnings('ignore')
        
        # Concept stats
        min_concepts = sys.maxsize
        max_concepts = 0
        total_concepts = 0
        concepts_distrib = {}
        
        # Define max score and current score
        max_score = len(gt_pairs)
        current_score = 0
        
        # Check there are the same number of pairs between candidate and ground truth
        if len(candidate_pairs) != len(gt_pairs):
            print('ERROR : Candidate does not contain the same number of entries as the ground truth!')
            exit(1)
        
        
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
    
    def tune_k(images_sims, data_dict, train_img_names, val_data):
        
        """
        This function:
        
        - Retrieves the k-number of training images with the highest cosine similarity for each validation images
        - Retrieves the most frequent concepts out of the k-number of training images for each image
        - gets the best K-number with the highest f1 score(max score)
        """
        
        # tune k for validation data based 
        best_k = 1
        max_score = 0
        for k in tqdm(range(1, 201)):
            val_results = {}
            for image_sim in images_sims:
                ###retreive k-training images(indices) with highest cosine similarity
                topk = np.argsort(images_sims[image_sim])[-k:] ##return indices
                concepts_list = []
                sum_concepts = 0
                for index in topk:
                    concepts = data_dict[train_img_names[index]]
                    sum_concepts += len(concepts)
                    for concept in concepts:
                        concepts_list.append(concept)
                ##return the most frequent concepts
                frequent_concepts = Counter(concepts_list).most_common(round(sum_concepts / k))
                val_results[image_sim] = ";".join(f[0] for f in frequent_concepts)
            ##calculate the f1 score for each k-number
            score = calculate_f1(val_results, val_data) ##calculate_f1(candidate_pairs, gt_pairs)
            if score > max_score:
                max_score = score
                best_k = k
        
        return max_score, best_k
    
    max_score, best_k = tune_k(img_sims, train_dict, img_names, val_dict)
    print("\n Found best f1 score on validation data:", max_score, "\n for k =", best_k)
    
    ##lets test the Best K on the Test Data and calculate the f1 score, get predictions
    def test_knn(df, image_vectors, best_k, data_dict, train_img_names, test_data):
        
        """
        :param best_k: Integer that represent the K distance.
        """
        
        base_model = DenseNet121(weights='imagenet', include_top=True)
        vector_extraction_model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        test_results = {}
        ##calculate test_images 'image embeddings'
        for i, test_image in tqdm(enumerate(df['image_name'])):
            image_path = df['image_path'][i]
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            vec = vector_extraction_model.predict(x).transpose().flatten()
            vec = vec / np.sum(vec)
            # clone to do efficient mat mul dot
            #compute cosine similarity btw test and train image vectors
            test_mat = np.array([vec] * image_vectors.shape[0])
            sims = np.sum(test_mat * image_vectors, 1)
            topk = np.argsort(sims)[-best_k:]
            concepts_list = []
            sum_concepts = 0
            for index in topk:
                concepts = data_dict[train_img_names[index]]
                sum_concepts += len(concepts)
                for concept in concepts:
                    concepts_list.append(concept)
            frequent_concepts = Counter(concepts_list).most_common(round(sum_concepts / best_k))
            test_results[test_image] = ";".join(f[0] for f in frequent_concepts)
        ##calculate the f1 score
        test_score = calculate_f1(test_results, test_data) ##calculate_f1(candidate_pairs, gt_pairs)
        
        return test_score, test_results
    
    test_f1score, test_preds = test_knn(df_test, img_vects, best_k, train_dict, img_names, test_dict)
    
    print('\n The F1 Score for the test data:', test_f1score, '\n')
    
    ###save the predicted and the ground truth concepts as .CSV files
    ##First, create dataframe for both predicted and groundtruth data
    df_gt = pd.DataFrame(test_dict.items(), columns = ['image_name', 'Ground_truth Concepts'])
    df_pred = pd.DataFrame(test_preds.items(), columns = ['image_name', 'Predicted Concepts'])
    #merge them
    df_gt_pred = pd.merge(df_gt, df_pred, on='image_name')
    
    ##save to csv
    df_gt_pred.to_csv(test_dir + 'groundtruth_predicted.csv', index = False)
    
###Arguments to pass to this Script
def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments")
    ###positional arguments
    ##file directories
    parser.add_argument('train_dir', type=str, help='dir that contains train images and cui.txt->(Concepts for the images)')
    parser.add_argument('val_dir', type=str, help='dir that contains validation images and cui.txt->(Concepts for the images)')
    parser.add_argument('test_dir', type=str, help='dir that contains test images and cui.txt->(Concepts for the images)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    knn_model(args.train_dir, args.val_dir, args.test_dir)