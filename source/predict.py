from __future__ import print_function # future proof
import argparse
import sys
import os
import json
import re 
import time

# pytorch
import torch
import torch.nn as nn

# torchtext
import torchtext
from torchtext.data import TabularDataset
from torchtext import data

# import model
from model import AdamNetV2
from utils import striphtml, tokenizer

def model_fn(model_dir):
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    txtField_path = os.path.join(model_dir, 'txt_field.pth')
    with open(txtField_path, 'rb') as f:
        txt_field = torch.load(f)
    labelField_path = os.path.join(model_dir, 'label_field.pth')
    with open(labelField_path, 'rb') as f:
        label_field = torch.load(f)
        
    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdamNetV2(vocab_size=model_info['vocab_size'],
                      embedding_dim=model_info['embedding_dim'], 
                      hidden_dim=model_info['hidden_dim'],
                      n_layers=model_info['n_layers'],
                      is_bidirectional=False,
                      dropout=model_info['dropout'],
                      output_dim=model_info['output_dim'],
                      padding_idx=model_info['padding_idx'],
                      txt_field = txt_field,
                      label_field = label_field
                      )
    
    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    return model.to(device)

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenized = tokenizer(input_data)
    
    numericalized = [model.txt_field.vocab.stoi[t] for t in tokenized] 
    sentence_length = torch.LongTensor([len(numericalized)]).to(device) 
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device) 
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        forwardpass = model(tensor, sentence_length)
    
    _, indices = torch.topk(torch.sigmoid(forwardpass),k=3)
    tags = [model.label_field.vocab.itos[t] for t in indices.tolist()[0]]
    return tags