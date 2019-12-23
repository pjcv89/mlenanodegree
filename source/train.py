from __future__ import print_function # future proof
import argparse
import sys
import os
import json
import re 
import time

import pandas as pd

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
#import torch.utils.data

# torchtext
import torchtext
from torchtext.data import TabularDataset, BucketIterator
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

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdamNetV2(model_info['embedding_dim'], 
                      model_info['hidden_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)

def accuracy(predictions, y): 
    correct = (torch.sigmoid(predictions).argmax(1)==y).sum()
    acc = correct.item() / len(y)
    return acc

def train_batch(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        (batchFeatures,batchLengths),batchLabels = batch
        predictions = model(batchFeatures,batchLengths).squeeze(1)
        loss = criterion(predictions, batchLabels)
        acc = accuracy(predictions, batchLabels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_batch(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            (batchFeatures,batchLengths),batchLabels = batch
            predictions = model(batchFeatures,batchLengths).squeeze(1)
            loss = criterion(predictions, batchLabels)
            acc = accuracy(predictions, batchLabels)
            
            epoch_loss += loss.item()
            epoch_acc += acc
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Load the training and validation data from a csv file
def get_train_valid_loader(batch_size, data_dir, device):
    
    txt_field = data.Field(sequential=True, 
                       tokenize=tokenizer, 
                       include_lengths=True, 
                       use_vocab=True)

    label_field = data.LabelField(dtype=torch.long)

    #train_val_fields = [('title', txt_field),('body', None),('tag', label_field)]
    train_val_fields = [('title', txt_field),('tags', label_field)]
    
    trainds, validds, testds = data.TabularDataset.splits(path=data_dir, 
                                            format='csv', 
                                            train='train.csv', 
                                            validation='valid.csv',
                                            test='test.csv', 
                                            fields=train_val_fields, 
                                            skip_header=True)
    
    txt_field.build_vocab(trainds,validds,testds,max_size=10000)
    label_field.build_vocab(trainds,validds,testds)
    
    train_iter = BucketIterator(trainds,batch_size=batch_size,device=device,sort_key=lambda x: len(x.title),sort_within_batch=True)
    valid_iter = BucketIterator(validds,batch_size=batch_size,device=device,sort_key=lambda x: len(x.title),sort_within_batch=True)
    test_iter = BucketIterator(testds,batch_size=batch_size,device=device,sort_key=lambda x: len(x.title),sort_within_batch=True)
    
    return train_iter, valid_iter, test_iter, txt_field, label_field

# Provided train function
def train(model, train_iter, valid_iter, epochs, optimizer, criterion, device):
    for epoch in range(epochs):  
        start_time = time.time()
        train_loss, train_acc = train_batch(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate_batch(model,valid_iter,criterion)
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    # save trained model, after all epochs
    save_model(model, args.model_dir)


# Provided model saving functions
def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # save state dictionary
    torch.save(model.cpu().state_dict(), path)
    
def save_model_params(model, txt_field, label_field, model_dir):
    
    vocab_size = len(txt_field.vocab)
    output_dim = len(label_field.vocab)
    
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'vocab_size': vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'n_layers': args.n_layers,
            'dropout': args.dropout,
            'padding_idx': args.padding_idx,
            'output_dim': output_dim
        }
        torch.save(model_info, f)
    
    txtField_path = os.path.join(args.model_dir, 'txt_field.pth')
    labelField_path = os.path.join(args.model_dir, 'label_field.pth')
    with open(txtField_path, 'wb') as f:
        torch.save(txt_field, f)
    with open(labelField_path, 'wb') as f:
        torch.save(label_field, f)
        


## TODO: Complete the main code
if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
  
    ## TODO: Add args for the three model parameters: input_dim, hidden_dim, output_dim
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=100, metavar='IN',
                        help='number of embedding dimensions (default: 100)')
    parser.add_argument('--hidden_dim', type=int, default=32, metavar='H',
                        help='hidden dim of model (default: 32)')
    parser.add_argument('--n_layers', type=int, default=2, metavar='NL',
                        help='number of lstm layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='DR',
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--padding_idx', type=int, default=1, metavar='PI',
                        help='padding index (default: 1)')

    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # get train loader
    train_iter, valid_iter, _, txt_field, label_field = get_train_valid_loader(args.batch_size, args.data_dir, device)
    
    ## TODO:  Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    #model = SimpleNet(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    
    vocab_size = len(txt_field.vocab)
    output_dim = len(label_field.vocab)
    
    model = AdamNetV2(vocab_size=vocab_size,
                  embedding_dim=args.embedding_dim,hidden_dim= args.hidden_dim,n_layers=args.n_layers,
                  dropout=args.dropout,padding_idx=1,
                  output_dim=output_dim,
                  is_bidirectional=False).to(device)
    
    # Given: save the parameters used to construct the model
    save_model_params(model, txt_field, label_field, args.model_dir)

    ## TODO: Define an optimizer and loss function for training
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Trains the model (given line of code, which calls the above training function)
    # This function *also* saves the model state dictionary
    #train(model, train_loader, args.epochs, optimizer, criterion, device)
    train(model, train_iter, valid_iter, args.epochs, optimizer, criterion, device)
    
    
