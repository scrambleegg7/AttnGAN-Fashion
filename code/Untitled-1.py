#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../Fashion'))
	print(os.getcwd())
except:
	pass

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import os, sys
import re
import string
import pathlib
import random
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


#%%
print('Python version:',sys.version)
print('Pandas version:',pd.__version__)
print('Pytorch version:', torch.__version__)
print('Spacy version:', spacy.__version__)


#%%
filename = 'SentimentAnalysisDataset.csv'
data_dir = "/home/donchan/Documents/DATA"
sens_file = os.path.join(data_dir,filename)

df = pd.read_csv(sens_file, error_bad_lines=False)
df.shape

#%% [markdown]
# ##  tokenizer 

#%%
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])

#%% [markdown]
# ## word2index vector / embedding vector

#%%

glove_dir = "/home/donchan/Documents/gloVe"
glove_50d = "glove.6B.50d.txt"
gloVe_file = os.path.join(glove_dir,glove_50d)

# setup wordVector from gloVe
words = []
idx = 0
word2idx = {}
vectors = []

batch_size = 2 

print("glove file setting.")
with open(gloVe_file, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

vectors_np = np.array(vectors[1:])
vectors_torch = torch.from_numpy(vectors_np)
print("to be loaded vectors size.", vectors_torch.size())

#%% [markdown]
# ## indexer 

#%%
def indexer(s): 
    wordindexes = []
    for w in nlp(s):
        try:
            ww = word2idx[w.text.lower()]
            wordindexes.append( ww )
        except:
            pass
                          
    return wordindexes


#%%
class VectorizeData(Dataset):
    def __init__(self, df_path, maxlen=10):
        self.maxlen = maxlen
        self.df = pd.read_csv(df_path, error_bad_lines=False)
        self.df = self.df[:10000] # select top10000
        
        self.df['SentimentText'] = self.df.SentimentText.apply(lambda x: x.strip())
        print('Indexing...')
        self.df['sentimentidx'] = self.df.SentimentText.progress_apply(indexer)
        print('Calculating lengths')
        self.df['lengths'] = self.df.sentimentidx.progress_apply(lambda x: self.maxlen if len(x) > self.maxlen else len(x))
        print('Padding')
        self.df['sentimentpadded'] = self.df.sentimentidx.progress_apply(self.pad_data)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        X = self.df.sentimentpadded[idx]
        lens = self.df.lengths[idx]
        y = self.df.Sentiment[idx]
        return X,y,lens
    
    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded


#%%
vectDataset = VectorizeData(sens_file, maxlen=128)


#%%
print(vectDataset[:4])


#%%
data_loader = DataLoader(vectDataset, batch_size=4)
print('Total batches', len(data_loader))

#%% [markdown]
# ## Model

#%%

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


#%%
vocab_size = len(words)
embedding_dim = 50
n_hidden = 100
n_out = 2


#%%
class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out):
        super().__init__()
        self.vocab_size,self.embedding_dim,self.n_hidden,self.n_out = vocab_size, embedding_dim, n_hidden, n_out
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        #self.gru = nn.GRU(self.embedding_dim, self.n_hidden)
        self.gru = nn.GRU(embedding_dim, n_hidden, batch_first=True)
        
        self.out = nn.Linear(self.n_hidden, self.n_out)
        
    def forward(self, seq, lengths, gpu=True):  # seq should be dim x batch_size
        #print('Sequence shape',seq.shape)
        #print('Lengths',lengths)
        bs = seq.size(0) # batch size
        #print('batch size', bs)
        self.h = self.init_hidden(bs, gpu) # initialize hidden state of GRU
        #print('Inititial hidden state shape', self.h.shape)
        embs = self.emb(seq)              
        #print("embedding(seq) = ", embs.shape)
        
        #embs = pack_padded_sequence(embs, lengths) # unpad
        #print("pack_padded_seq = ", embs)
        
        
        gru_out, self.h = self.gru(embs, self.h) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        
        #print("gru and hidden size after gru", gru_out.shape, self.h.shape)
        
        #gru_out, lengths = pad_packed_sequence(gru_out) # pad the sequence to the max length in the batch
        #print('GRU output(all timesteps)', gru_out.shape)  
        #print(gru_out)
        #print('GRU last timestep output')
        #print(gru_out[-1])
        #print('Last hidden state', self.h.shape)
        last_layer = self.h.squeeze(0)
        # since it is as classification problem, we will grab the last hidden state
        outp = self.out(last_layer) # self.h[-1] contains hidden state of last timestep
        return F.log_softmax(outp, dim=-1)
    
    def init_hidden(self, batch_size, gpu):
        if gpu: return Variable(torch.zeros((1,batch_size,self.n_hidden)).cuda())
        else: return Variable(torch.zeros((1,batch_size,self.n_hidden)))


#%%

m = SimpleGRU(vocab_size, embedding_dim, n_hidden, n_out)


#%%
print(m)

#%% [markdown]
# ## Sort batch

#%%
def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths
    #return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)


#%%
X, y, lens = next( iter( data_loader ))


#%%
print( X.shape, y.shape, lens)
X, y, lens = sort_batch(X, y, lens)
print( X.shape, y.shape, lens)


#%%
outp = m(X,lens.cpu().numpy(), gpu=False)


#%%
outp


#%%
torch.max(outp, dim=1)


#%%
print(y.shape)
print(y)


#%%
F.nll_loss(outp, Variable(y))


#%%
r = torch.randn(1,4,100)
print(r.shape)
print( r.view(4,100).shape )
print( r.squeeze(0).shape ) 

#%% [markdown]
# # Training after putting all necesary staffs together

#%%
vocab_size = len(words)
embedding_dim = 10
n_hidden = 5
n_out = 2


#%%
class SimpleGRU_v2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out):
        super().__init__()
        self.vocab_size,self.embedding_dim,self.n_hidden,self.n_out = vocab_size, embedding_dim, n_hidden, n_out
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden, batch_first=True)
        self.out = nn.Linear(self.n_hidden, self.n_out)
        
    def forward(self, seq, lengths, gpu=True):
        bs = seq.size(0) # batch size
        self.h = self.init_hidden(bs, gpu) # initialize hidden state of GRU
        embs = self.emb(seq)
        #embs = pack_padded_sequence(embs, lengths) # unpad
        gru_out, self.h = self.gru(embs, self.h) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        #gru_out, lengths = pad_packed_sequence(gru_out) # pad the sequence to the max length in the batch
        # since it is as classification problem, we will grab the last hidden state
        outp = self.out(self.h[-1]) # self.h[-1] contains hidden state of last timestep
        return F.log_softmax(outp, dim=-1)
    
    def init_hidden(self, batch_size, gpu):
        if gpu: return Variable(torch.zeros((1,batch_size,self.n_hidden)).cuda())
        else: return Variable(torch.zeros((1,batch_size,self.n_hidden)))


#%%
class SimpleGRU_v3(nn.Module):
    def __init__(self, weights_matrix, vocab_size, embedding_dim, n_hidden, n_out):
        super().__init__()
        
        self.n_hidden,self.n_out = n_hidden, n_out
        self.emb, num_embeddings, self.embedding_dim = create_emb_layer(weights_matrix, False)
        
        #self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden, batch_first=True)
        self.out = nn.Linear(self.n_hidden, self.n_out)
        
    def forward(self, seq, lengths, gpu=True):
        bs = seq.size(0) # batch size
        self.h = self.init_hidden(bs, gpu) # initialize hidden state of GRU
        embs = self.emb(seq)
        #embs = pack_padded_sequence(embs, lengths) # unpad
        gru_out, self.h = self.gru(embs, self.h) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        #gru_out, lengths = pad_packed_sequence(gru_out) # pad the sequence to the max length in the batch
        # since it is as classification problem, we will grab the last hidden state
        outp = self.out(self.h[-1]) # self.h[-1] contains hidden state of last timestep
        return F.log_softmax(outp, dim=-1)
    
    def init_hidden(self, batch_size, gpu):
        if gpu: return Variable(torch.zeros((1,batch_size,self.n_hidden)).cuda())
        else: return Variable(torch.zeros((1,batch_size,self.n_hidden)))

#%% [markdown]
# # V2 training 

#%%
#rain_dl = DataLoader(ds, batch_size=512)
train_dl = DataLoader(vectDataset, batch_size=4)
m = SimpleGRU_v2(vocab_size, embedding_dim, n_hidden, n_out).cuda()
opt = optim.Adam(m.parameters(), 1e-2)


#%%
def fit(model, train_dl, val_dl, loss_fn, opt, epochs=3):
    num_batch = len(train_dl)
    for epoch in tnrange(epochs):      
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0
        
        if val_dl:
            y_true_val = list()
            y_pred_val = list()
            total_loss_val = 0
        
        t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
        for X,y, lengths in t:
            t.set_description(f'Epoch {epoch}')
            X,y,lengths = sort_batch(X,y,lengths)
            X = Variable(X.cuda())
            y = Variable(y.cuda())
            lengths = lengths.numpy()
            
            opt.zero_grad()
            pred = model(X, lengths, gpu=True)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            
            t.set_postfix(loss=loss.data.item())
            pred_idx = torch.max(pred, dim=1)[1]
            
            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(pred_idx.cpu().data.numpy())
            total_loss_train += loss.data.item()
            
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_loss = total_loss_train/len(train_dl)
        print(f' Epoch {epoch}: Train loss: {train_loss} acc: {train_acc}')




#%%
fit(model=m, train_dl=train_dl, val_dl=None, loss_fn=F.nll_loss, opt=opt, epochs=10)

#%% [markdown]
# # V3 Training

#%%
#rain_dl = DataLoader(ds, batch_size=512)

# embedding fixed with pretrained model
# so that, all training parameters are ingnored.
# loss / acc are not implemented during training session

train_dl = DataLoader(vectDataset, batch_size=4)
m3 = SimpleGRU_v3( vectors_torch,  vocab_size, embedding_dim, 100, n_out).cuda()
opt = optim.Adam(m.parameters(), 1e-2)


#%%
fit(model=m3, train_dl=train_dl, val_dl=None, loss_fn=F.nll_loss, opt=opt, epochs=10)


#%%



