from torchtext import data
#from torchtext.data import DataSets
from torchtext import datasets
from torchtext.vocab import GloVe

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable

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
from PIL import Image

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from collections import defaultdict

from scipy.misc import imsave

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file


glove_dir = "/home/donchan/Documents/gloVe"
glove_50d = "glove.6B.50d.txt"
gloVe_file = os.path.join(glove_dir,glove_50d)


def read_data(file_name):
    with open(file_name,'r') as f:
        word_vocab = set() # not using list to avoid duplicate entry
        word2vector = {}
        for line in f:
            line_ = line.strip() #Remove white space
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
    print("Total Words in DataSet:",len(word_vocab))
    return word_vocab,word2vector


def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys]

def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Scale(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret

class FashionTextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=256,
                 transform=None, target_transform=None):
        
        self.transform = transform

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2


        self.tokenizer = RegexpTokenizer(r'\w+')       

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('CelebA') != -1:
            #self.bbox = self.load_bbox()
            self.bbox = None
        else:
            self.bbox = None
        #split_dir = os.path.join(data_dir, split)

        #if split == "train":
        filepath = os.path.join(data_dir,"fashiongen_256_256_train.h5")
        self.data_h5py = h5py.File(filepath, "r")

        filepath_valid = os.path.join(data_dir,"fashiongen_256_256_validation.h5")
        self.data_h5py_valid = h5py.File(filepath_valid, "r")
    
        self.split_dir = split

        if split == "train":
            group_keys = list(self.data_h5py.keys())
            print(group_keys)
            self.indexes =  self.data_h5py["index"]
            #print("total data length",len(self.indexes))
            self.number_example = len(self.indexes)

        if split == "valid":
            group_keys = list(self.data_h5py_valid.keys())
            print(group_keys)
            self.indexes =  self.data_h5py_valid["index"]
            #print("total data length",len(self.indexes))
            self.number_example = len(self.indexes)


        self.categories = []
        self.image_indexes = []
        self.class_id = []

        self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data()
        #self.filenames, self.captions, self.ixtoword, \
        #    self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        #self.class_id = self.load_class_id(split_dir, self.number_example)


        print("result from Dataset initialization.")
        print("length of captions",len(self.captions))
        print("length of ixtoword", len(self.ixtoword))
        print("length of wordtoix", len(self.wordtoix))
        print("number of words", self.n_words)
        
    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'Anno/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        print (df_bounding_boxes)
        #
        filepath = os.path.join(data_dir, 'Anno/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox


    def build_word_dictionary(self):
        word_counts = defaultdict(float)

        print("Building Words Dictionary...........")
        start_t = time() 
        caption_grp_name = "input_concat_description"
        captions = self.data_h5py[caption_grp_name]
        captions_valid = self.data_h5py_valid[caption_grp_name]
        
        #lengthOfCaptions = len(captions)
        lengthOfCaptions = 10000
        print("length of captions", lengthOfCaptions )
        
        train_captions = []
        valid_captions = []

        stop_words = stopwords.words('english')
        
 
        for indx in range( lengthOfCaptions ):

            text = captions[indx][0].decode('cp437')
            reg_token_words = self.tokenizer.tokenize(text)
            reg_token_words = [w.lower() for w in reg_token_words  if w.isalpha() ]

            words_filtered = reg_token_words[:] # creating a copy of the words list
            for word in reg_token_words:
               if word in stop_words:        
                    words_filtered.remove(word)

            # append text by category 
            train_captions.append( words_filtered )

        #lengthOfCaptions = len(captions_valid)
        for indx in range( lengthOfCaptions ):

            text_valid = captions_valid[indx][0].decode('cp437')
            reg_token_words_valid = self.tokenizer.tokenize(text_valid)
            reg_token_words_valid = [ w.lower() for w in reg_token_words_valid if w.isalpha() ]

            words_filtered = reg_token_words_valid[:] # creating a copy of the words list
            for word in reg_token_words_valid:
               if word in stop_words:        
                    words_filtered.remove(word)

            # append text by category 
            valid_captions.append( words_filtered )


        captions = train_captions + valid_captions
        print("total captions (train+validation)", len(captions) )
        
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in valid_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        print( "words counts = train + validation", len(ixtoword) )
        print("time to build words dictionary", time() - start_t )

        return train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)


    def build_dictionary(self):
        #word_counts = defaultdict(float)

        print("Building Dictionary...........")
        #start_t = time() 

        caption_grp_name = "input_concat_description"
        index_name = "index"
        category_name = "input_category"
        product_id_name = "input_productID"

        if self.split_dir == "train":
            indexes = self.data_h5py[index_name]
            categories = self.data_h5py[category_name]
            product_id = self.data_h5py[product_id_name]
            captions = self.data_h5py[caption_grp_name]
        if self.split_dir == "valid":
            indexes = self.data_h5py_valid[index_name]
            categories = self.data_h5py_valid[category_name]
            product_id = self.data_h5py_valid[product_id_name]
            captions = self.data_h5py_valid[caption_grp_name]

        #lengthOfCaptions = len(captions)
        lengthOfCaptions = 10000
        print("length of captions", lengthOfCaptions )
        
        #text = self.fashion_data["input_concat_description"][index][0].decode('cp437')

        max_lengthOfWords = 0
        train_captions = []


        for indx in range( lengthOfCaptions ):

            # save relative image filename
            filename = str( product_id[indx][0] )  +  "_"  + str( indexes[indx][0] ) + ".jpg"
            category = categories[ indx ][0].decode('cp437')

            filename = os.path.join( category, filename )
            self.image_indexes.append( filename )


            str_class_id = str( product_id[indx][0] )  + str( indexes[indx][0] )
            int_class_id = int( str_class_id )
            self.class_id.append( int_class_id )

            text = captions[indx][0].decode('cp437')
            reg_token_words = self.tokenizer.tokenize(text)
            reg_token_words = [w.lower() for w in reg_token_words ]

            # append text by category 
            train_captions.append( reg_token_words )

            lengthOfWords = len(reg_token_words)
            if lengthOfWords > max_lengthOfWords:
                max_lengthOfWords = lengthOfWords

            category = categories[ indx ][0].decode('cp437')
            self.categories.append( category )

            if indx % 100000 == 0:
                print("- " * 20)
                print("** TEXT:")
                print(text)
                print("** CATEGORY:")
                print(category)


        train_captions_new, valid_captions_new, ixtoword, wordtoix, length_ixtoword = self.build_word_dictionary()

        if self.split_dir == "train":
            return train_captions_new, ixtoword, wordtoix, length_ixtoword
        if self.split_dir == "valid":
            return valid_captions_new, ixtoword, wordtoix, length_ixtoword


    def load_text_data(self):

        captions, ixtoword, wordtoix, n_words = self.build_dictionary()


        return captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]  # select top 18 words from shuffle array.
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        #print("index from getitem function",index)

        filename = self.image_indexes[index]
        cls_id = self.class_id[index]


        #
        #if self.bbox is not None:
        #    bbox = self.bbox[key]
        #    data_dir = self.data_dir
        #else:
        bbox = None
        #data_dir = self.data_dir
        
        img = os.path.join( self.data_dir,  self.split_dir,  filename )
        #
        #key = os.path.basename(key)
        #key = key.split(".")[0]
        #img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img, self.imsize, bbox, self.transform, normalize=self.norm)
        # random select a sentence
        #sent_ix = random.randint(0, self.embeddings_num)
        #new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(index)
        return imgs, caps, cap_len, cls_id, filename


    def __len__(self):
        #return self.number_example
        return int(10000)





