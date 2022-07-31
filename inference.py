import os

INPUT_DIR = '../input/us-patent-phrase-to-phrase-matching/'
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class CFG:
    num_workers=2
    batch_size=16
    max_len=133
    seed=42
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]

import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
import scipy as sp
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

test = pd.read_csv(INPUT_DIR+'test.csv')
submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')
cpc_texts = torch.load('../input/pppm-deberta-v3-large-baseline-w-w-b-train/cpc_texts.pth')
test['context_text'] = test['context'].map(cpc_texts)
test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
CFG.tokenizer = AutoTokenizer.from_pretrained('../input/deberta-v3-large/deberta-v3-large')

def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs

# Model

class CNNModel(nn.Module):
    def __init__(self, path):
        super().__init__()

        config = AutoConfig.from_pretrained(path)
        config.update({"output_hidden_states":True, 
                "hidden_dropout_prob": 0.0,
                'return_dict':True})                      
        
        self.roberta = AutoModel.from_pretrained(path, config=config)
            
        self.conv1 = nn.Conv1d(config.hidden_size*2, 1024, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(1024, 512, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(512, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=4, batch_first=True)
        self.fc = nn.Linear(2048, 1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
    def forward(self, inputs):
        output = self.roberta(**inputs)        
        hs = output.hidden_states
        x = torch.cat([hs[-1], hs[-2]], -1)
        x = self.encoder_layer(x)
        conv1_logits = self.conv1(x.transpose(1, 2))
        conv2_logits = self.conv2(conv1_logits)
        conv3_logits = self.conv3(conv2_logits)
        x = conv3_logits.transpose(1, 2)
        x = torch.mean(x, 1)
        return x

def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG.batch_size,
                         shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
predictions = []
MMscaler = MinMaxScaler()
for fold in CFG.trn_fold:
    model = CNNModel('../input/deberta-v3-large/deberta-v3-large')
    state = torch.load(f'../input/download-models-from-huggingface1/ppp1/microsoft-deberta-v3-large_fold{fold}_best.pth',
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    #prediction = update_label(prediction)
    prediction = MMscaler.fit_transform(prediction)
    #print(prediction.shape)
    predictions.append(prediction)
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()
    #break
predictions = np.mean(predictions, axis=0)
submission['score'] = predictions
submission[['id', 'score']].to_csv('sub1.csv', index=False)