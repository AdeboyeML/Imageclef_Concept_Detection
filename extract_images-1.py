import os
import pandas as pd
import numpy as np
import re
import argparse

import shutil
import sys
import tarfile



def extract_images(dir_links):
    
    """
    Make sure to run this function in the (command line) directory you downloaded the tarfiles
    and in the directory that contains the 'dlinks.txt'
    
    This function perform the following tasks:
    -- Extract images from the tar files
    -- Rename the images to their respective "Roco Names"
    -- Transfer the images to the "images" folder
    -- Delete all temporary directories used for the image extractions
    """
    
    ##prepare the dataframe that contains the pmc_id column to be used to tar extraction
    df = pd.read_csv(dir_links, delimiter = '\t', header = None)
    df['url_list'] = df[1].apply(lambda x: x.split()[2])
    df['pmc_id'] = df['url_list'].apply(lambda x: x.split('/')[-1][:-7])
    df['img'] = df['pmc_id'] + '/' + df[2]
    df.sort_values(by = 'pmc_id', inplace = True)
    df.reset_index(inplace = True)
    df.drop(['index'], axis = 1, inplace = True)
    
    ##get the roco_data -- tar files and sort it
    roco_data = os.listdir('roco_data')
    roco_data.sort()
    
    
    ##function used to remove non-downloadable  or non-extracted files from the df
    def remove_missing_data(df, column, roco_list, id):
        
        pmc_id = df[column].values.tolist()
        
        def missing_index(pmc_id, roco_list, id):
            idx_missing = []
            for idx, x in enumerate(pmc_id[:-id]):
                if x != roco_list[idx]:
                    idx_missing.append(idx)
            return idx_missing, pmc_id
        
        for x in range(id):
            miss_id, pmc_id = missing_index(pmc_id, roco_list, id)
            id_miss = miss_id[0]
            del pmc_id[id_miss]
            df.drop([id_miss], inplace = True)
            df.reset_index(inplace = True)
            df.drop(['index'], axis = 1, inplace = True)
            id -= 1
        return df
    
    ##check for missing ids and remove them from the dataframe
    roco_list = [re.sub(r'\.tar.*', '', x) for x in roco_data]
    id = df.shape[0] - len(roco_data)
    df = remove_missing_data(df, 'pmc_id', roco_list, id)
    
    
    ##extract_tarfiles dir created
    os.mkdir('extract_tarfiles')
    
    ##Extract files with the specific images needed from the downloaded tar files
    idx_list = [] ##index of the tar files that do not contain images to be extracted
    ##extract images from the downloaded tar files
    for idx, x in enumerate(roco_data):
        image = df['img'][idx]
        archive_tarfile = tarfile.open(os.path.join('./roco_data', x))
        try:
            archive_tarfile.extract(image, 'extract_tarfiles')
            print(image, x)
        except:
            print('Could not extract images from {0}...'.format(x))
            idx_list.append(idx)
            continue
    
    ##drop the tar files in df that do not contain images to be extracted
    df.drop(idx_list, inplace = True)
    df.reset_index(inplace = True)
    df.drop(['index'], axis = 1, inplace = True)
    
    ##delete tar files in roco_data dir that do not contain images to be extracted
    for idx in idx_list:
        print('tar file to delete: {0}'.format(roco_data[idx]))
        os.unlink('roco_data/' + roco_data[idx])
        
    print('\n Amount of files, where images were successfully extracted from {0}'.format(df.shape[0]))
    
    ##load the extracted images into a specific folder -- "images_folder2"
    os.mkdir('images_folder2')
    for ext_file in os.listdir('./extract_tarfiles'):
        for images in os.listdir('./extract_tarfiles/' + ext_file):
            imgs = os.path.join('./extract_tarfiles/' + ext_file, images)
            shutil.copy(imgs, 'images_folder2')
            #print(imgs)
    print('\n Finished loading images into images_folder2 \n')
    
    img_list = os.listdir('images_folder2')
    print('\n Amount of images successfully extracted == {0} \n'.format(len(img_list)))
    
    ##amount of missing images after extraction 
    id_img = df.shape[0] - len(img_list)
    
    ###lets remove missing images from the df_tr
    df = remove_missing_data(df, 2, img_list, id_img)
    
    roco_names = df[0].values.tolist()
    ##rename the images to their "ROCO" names
    for idx, images in enumerate(img_list):
        print(images)
        os.rename('./images_folder2/' + images, './images_folder2/' + roco_names[idx] + '.jpg')
        
    ##check if the images folder already exist...if not create a new one
    if not os.path.exists('images'):
        os.mkdir('images')
        
    ###Finally,  lets copy the images to the original "images" folder ----
    print('\n Copying the new extracted images into the "images" folder \n')
    for images in os.listdir('images_folder2'):
        imgs = os.path.join('images_folder2/', images)
        shutil.copy(imgs, 'images')
        print(imgs)
        
    img_list = os.listdir('images')
    print('\n Amount of images in the images folder: {0} \n'.format(len(img_list)))
    
    ###delete the previous/initial files and folders created  --- they consume a large space
    folder_list = ['extract_tarfiles', 'images_folder2']
    for item in folder_list:
        try:
            shutil.rmtree(item)
        except:
            print('{0} already deleted'.format(item))
            continue
        else:
            print('Deleted {0}'.format(item))


def parse_args():
    
    parser = argparse.ArgumentParser(description="Parse arguments")
    ###positional argument
    parser.add_argument('input_dir', type=str, help='function to call')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    extract_images(args.input_dir)