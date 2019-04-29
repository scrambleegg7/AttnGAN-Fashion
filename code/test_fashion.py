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

import os
import numpy as np
from glob import glob

from nltk.tokenize import RegexpTokenizer

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from FashionTextDataset import FashionTextDataset 

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file


def main():

    cfg_from_file("./cfg/DAMSM/Fashion.yml")

    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    print("default imsize",imsize)
    print("batch size", batch_size)




    dataset = FashionTextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    imgs, caps, cap_len, cls_id, filename = next( iter( dataloader ) )
    print( imgs[0].shape)
    print( caps )
    print( cap_len )
    print( cls_id )
    print( filename )

    for caption in caps:
        words = []
        for i in range(18):
            ix = caption[i,0].data.numpy()
            w = dataset.ixtoword[ int(ix) ]
            words.append( w )
        print( words )
    

if __name__ == "__main__":
    main()