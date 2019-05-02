from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
#from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

#from datasets import TextDataset
from FashionTextDataset import prepare_data

from model import RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from FashionTextDataset import FashionTextDataset 
from datasets import TextDataset

import pickle

from GlobalAttention import func_attention

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


UPDATE_INTERVAL = 100
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def trainSingle(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()

    data = next( iter( dataloader ) )

        # print('step', step)
    rnn_model.zero_grad()
    cnn_model.zero_grad()

    imgs, captions, cap_lens, \
        class_ids, keys = prepare_data(data)


    # words_features: batch_size x nef x 17 x 17
    # sent_code: batch_size x nef
    words_features, sent_code = cnn_model(imgs[-1])
    #print("words features shape",words_features.shape)
    #print("sent code shape", sent_code.shape)
    # --> batch_size x nef x 17*17
    nef, att_sze = words_features.size(1), words_features.size(2)
    #print("nef att_sze", nef, att_sze)
    # words_features = words_features.view(batch_size, nef, -1)

    hidden = rnn_model.init_hidden(batch_size)
    # words_emb: batch_size x nef x seq_len
    # sent_emb: batch_size x nef


    #print("train captions", captions.size() )
    #print("train cap_lens", cap_lens.size() )
    #print("train word features", words_features.size() )
    #print("train sent_code", sent_code.size() )



    words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)
    #print("words_emb shape", words_emb.size() )
    #print("sent_emb shape", sent_emb.size() )


    words_loss(words_features, words_emb, labels, cap_lens, class_ids, batch_size)

    


    

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def build_models(dataset, batch_size):
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch




def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):


    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        
        print(class_ids,class_ids[i],masks)
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)

        #print("word shape" , word.size() )

        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)



    similarities = torch.cat(similarities, 1)
    print("** similarities **")
    print(similarities)
    print("")
    print("** class ids **")
    print(class_ids)
    print("")
    print("** labels **")
    print(labels)

    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    print(similarities)
    print(similarities.size())


    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None

    print("loss0 loss1")
    print(loss0.item(), loss1.item())

def testproc():

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset = FashionTextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)


    #dataset = FashionTextDataset(cfg.DATA_DIR, 'train',
    #                      base_size=cfg.TREE.BASE_SIZE,
    #                      transform=image_transform)


    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))

    imgs, caps, cap_len, cls_id, key = next( iter( dataloader ) )
    print( imgs[0].shape)

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models(dataset,batch_size)
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR
        for epoch in range(start_epoch, 1):
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            epoch_start_time = time.time()
            count = trainSingle(dataloader, image_encoder, text_encoder,
                          batch_size, labels, optimizer, epoch,
                          dataset.ixtoword, image_dir)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def testproc2():
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    mkdir_p(model_dir)
    mkdir_p(image_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])


    dataset = FashionTextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)


    print(dataset.n_words, dataset.embeddings_num)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=True, num_workers=int(cfg.WORKERS))


    
    # Train ##############################################################
    rnn_model, cnn_model, labels, start_epoch = build_models()
    para = list(rnn_model.parameters())
    for v in cnn_model.parameters():
        if v.requires_grad:
            para.append(v)

    data = next( iter( dataloader ) )
    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

    
    # check last item from images list. 
    print( imgs[-1].shape)
    print( "labels", labels)
    #print("class ids",class_ids)

    words_features, sent_code = cnn_model(imgs[-1])
    print("words features shape",words_features.shape)
    print("sent code shape", sent_code.shape)
    # --> batch_size x nef x 17*17
    nef, att_sze = words_features.size(1), words_features.size(2)
    print("nef att_sze", nef, att_sze)
    # words_features = words_features.view(batch_size, nef, -1)

    hidden = rnn_model.init_hidden(batch_size)
    for i, h in enumerate( hidden ):
        print("hidden size",i+1,  h.size() )

    # 2 x batch_size x hidden_size
    
    # words_emb: batch_size x nef x seq_len
    # sent_emb: batch_size x nef


    print("train captions", captions.size() )
    print("train cap_lens", cap_lens.size() )
    #print("train word features", words_features.size() )
    #print("train sent_code", sent_code.size() )

    words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)
    print("words_emb shape", words_emb.size() )
    print("sent_emb shape", sent_emb.size() )


    i = 10
    masks = []
    if class_ids is not None:
        mask = (class_ids == class_ids[i]).astype(np.uint8)
        mask[i] = 0
        masks.append(mask.reshape((1, -1)))

    print("no masks, if class ids are sequential.", masks)


    #data_dir = "/home/donchan/Documents/DATA/CULTECH_BIRDS/CUB_200_2011/train"
    #if os.path.isfile(data_dir + '/class_info.pickle'):
    #    with open(data_dir + '/class_info.pickle', 'rb') as f:
    #        class_id = pickle.load(f, encoding="latin1")



    # Get the i-th text description
    words_num = cap_lens[i]
    print(words_num)
    # -> 1 x nef x words_num
    word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
    print(word.size())
    # -> batch_size x nef x words_num
    word = word.repeat(batch_size, 1, 1)
    print(word.size())
    #print(word)

    context = words_features.clone()
    query = word.clone()

    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)

    #print("attn on Eq.8 on GlobalAttention", attn.size()  , attn.data.cpu().sum() ) # 13872, 6   / 13872, 7 ??
    attn = nn.Softmax(dim=0)(attn)  # Eq. (8)
    print("attn size", attn.size())

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    print("attn size", attn.size())
    
    #print("attn on Eq.9 on GlobalAttention", attn.size() , attn.data.cpu().sum() ) # 288, 289 / 336 , 289 ?
    
    #  Eq. (9)
    
    attn = attn * cfg.TRAIN.SMOOTH.GAMMA1
    attn = nn.Softmax(dim=0)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)
    print("weight size", weightedContext.size())

    attn = attn.view(batch_size, -1, ih, iw)
    print("attn size after Eq9", attn.size())

    att_maps = []
    #weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
    att_maps.append(attn[i].unsqueeze(0).contiguous())
    # --> batch_size x words_num x nef
    word = word.transpose(1, 2).contiguous()
    weightedContext = weightedContext.transpose(1, 2).contiguous()
    # --> batch_size*words_num x nef
    word = word.view(batch_size * words_num, -1)
    weightedContext = weightedContext.view(batch_size * words_num, -1)
    print("weight size after Eq.10", weightedContext.size())

    #
    # -->batch_size*words_num
    row_sim = cosine_similarity(word, weightedContext)
    print("row similarities", row_sim.size())
    # --> batch_size x words_num
    row_sim = row_sim.view(batch_size, words_num)

    # Eq. (10)
    row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
    row_sim = row_sim.sum(dim=1, keepdim=True)
    row_sim = torch.log(row_sim)

    print(row_sim)
    # --> 1 x batch_size
    # similarities(i, j): the similarity between the i-th image and the j-th text description
    #similarities.append(row_sim)


def main():
    testproc()

if __name__ == "__main__":
    main()