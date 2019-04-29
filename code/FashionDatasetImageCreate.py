

import sys

import h5py
import numpy as hp
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

import os, errno
import numpy as np
from glob import glob
import pandas as pd  

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

from scipy.misc import imsave

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

class FashionDatasetImageCreate(Dataset):

    def __init__(self, vocab_index=None):
    
        self.tokenizer = RegexpTokenizer(r'\w+')        


    def createDataSetImage(self, data_dir = "data" , mode="train"):

        if mode == "train":
            filepath = os.path.join(data_dir,"fashiongen_256_256_train.h5")
            image_data_dir = os.path.join(  data_dir, "train" )
            #image_data_dir = "/home/donchan/Documents/DATA/FashionGAN/train"
        else:
            filepath = os.path.join(data_dir,"fashiongen_256_256_validation.h5")
            image_data_dir = os.path.join(  data_dir, "valid" )
            #image_data_dir = "/home/donchan/Documents/DATA/FashionGAN/valid"


        self.data_h5py = h5py.File(filepath, "r")

        caption_grp_name = "input_concat_description"
        index_name = "index"
        category_name = "input_category"
        image_name = "input_image"
        product_id_name = "input_productID"

        indexes = self.data_h5py[index_name]
        captions = self.data_h5py[caption_grp_name]
        categories = self.data_h5py[category_name]
        images = self.data_h5py[image_name]
        product_id = self.data_h5py[product_id_name]


        self.text_array = []
        filecreateSkipCounter = 0

        for i in range(  len(  captions )  ):
            self.text = captions[i][0].decode('cp437')
            category = categories[ i ][0].decode('cp437')
            
            #self.text_array.append(  self.text )

            filename = str( product_id[i][0] )  +  "_"  + str( indexes[i][0] ) + ".jpg"
            filename = os.path.join( image_data_dir, category , filename )
            target_dir = os.path.join( image_data_dir, category )

            try:
                os.makedirs(target_dir)
            except OSError as ex:
                if ex.errno == errno.EEXIST and os.path.isdir(target_dir):
                    # ignore existing directory
                    pass
                else:
                    # a different error happened
                    raise

            if os.path.exists(filename):
                #print("filename overlapped", filename)
                filecreateSkipCounter += 1
                if filecreateSkipCounter % 1000 == 0:
                    print("file creation skip counter is over %d "  % filecreateSkipCounter)
                continue
                
            if i % 1000 == 0:
                print("%d files saved....." % i)
                print(filename)
            imsave(  filename, images[i]  )



    def __len__(self):
        return len(self.data_indexes)
            
    def __getitem__(self, index):
    
        
        #text = self.fashion_data["input_concat_description"][index][0].decode('cp437')

        text = self.text_array[index]
        #image = self.images[index]

        return text


def main():

    

    FashionDataDir = "/home/donchan/Documents/DATA/FashionGAN"
    #filepath = os.path.join(FashionDataDir,"fashiongen_256_256_train.h5")
    #fashion_data = h5py.File(filepath, "r")


    #index = 20
    #img = fashion_data["input_image"][index]

    #print(img.shape)


    fdataset = FashionDatasetImageCreate()

    fdataset.createDataSetImage(FashionDataDir, "train")
    fdataset.createDataSetImage(FashionDataDir, "valid")
    

if __name__ == "__main__":
    main()
