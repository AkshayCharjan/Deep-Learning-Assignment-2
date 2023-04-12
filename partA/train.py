import pytorch_lightning as pl
import pandas as pd
import os 
import torch
import torchvision
from torch.utils.data import Dataset ,DataLoader
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import torch
from sklearn.model_selection import train_test_split 
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models


import argparse
import wandb


wandb.login(key="a2e6402ce9fe2ebe1f01d5332c4fafa210b0dc0c")
pName = "Assignment 2 Part A main py"
run_obj=wandb.init( project=pName)


parser = argparse.ArgumentParser()
parser.add_argument('-wp','--wandb_project',default ='myprojectname',metavar="",required = False,type=str,help = "Project name used to track experiments in Weights & Biases dashboard" )
parser.add_argument('-we','--wandb_entity',default ='myname',metavar="",required = False,type=str,help = "Wandb Entity used to track experiments in the Weights & Biases dashboard." )
parser.add_argument('-e','--epochs',default=10,metavar="",required = False,type=int,help = "Number of epochs to train neural network.")
parser.add_argument('-do','--drop_out',default=0.3,metavar="",required = False,type=float,help = "Dropout")
parser.add_argument('-lr','--learning_rate',default=0.0001,metavar="",required = False,type=float,help = "Learning rate used to optimize model parameters")
parser.add_argument('-a','--activation_function',default='GELU',metavar="",required = False, help = "Activation Function", type=str,choices= ["SiLU", "Mish", "GELU", "ReLU"])
parser.add_argument('-bn','--batch_normalization',default='No',metavar="",required = False,type=str, help = "batch normalization", choices= ["Yes", "No"])
parser.add_argument('-da','--data_augmentation',default='No',metavar="",required = False, type=str,help = "data augmentation", choices= ["Yes", "No"])
parser.add_argument('-fz','--filter_size',default=64,metavar="",required = False, type=int,help = "filter_size")
parser.add_argument('-fo','--filter_organisation',default="same",metavar="",required = False,type=str, help = "filter_organisation", choices= ["same","half","double"])

args=parser.parse_args()

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((256,256)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

transform_augmented = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.AutoAugment(),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


# Load dataset from directory
if(args.data_augmentation=="No"):
    dataset = datasets.ImageFolder('inaturalist_12K/train', transform=transform)
else:
    dataset = datasets.ImageFolder('/content/inaturalist_12K/train', transform=transform_augmented)

test_dataset = datasets.ImageFolder('/content/inaturalist_12K/val', transform=transform)



# Split dataset into training and testing sets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

# Create data loader objects for training and testing sets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)


class CNNModel(pl.LightningModule):
    
    def __init__(self,activation_function,batch_normalization,data_augmentation,filter_organisation,drop_out):
        self.activation_function=activation_function
        self.batch_normalization=batch_normalization

        super(CNNModel, self).__init__()
        
        self.cnv1 = torch.nn.Conv2d(3, filter_organisation[0], 3)
        self.cnv2 = torch.nn.Conv2d(filter_organisation[0], filter_organisation[1], 3)
        self.cnv3 = torch.nn.Conv2d(filter_organisation[1], filter_organisation[2], 3)      
        self.cnv4 = torch.nn.Conv2d(filter_organisation[2], filter_organisation[3], 3)
        self.cnv5 = torch.nn.Conv2d(filter_organisation[3], filter_organisation[4], 3)
        
        if(activation_function=="ReLU"):
            self.activation_function=nn.ReLU()
        elif(activation_function=="GELU"):
            self.activation_function=nn.GELU()
        elif(activation_function=="SiLU"):
            self.activation_function=nn.SiLU()
        elif(activation_function=="Mish"):
            self.activation_function=nn.Mish()
        
        stride=2
        input_size=256
        
        DenseLayerSize=input_size
        for filter in filter_organisation:
            DenseLayerSize = (DenseLayerSize-4)//stride + 1
        
        self.bn = nn.BatchNorm1d(DenseLayerSize*DenseLayerSize*filter_organisation[4])
        self.mxpool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc= nn.Linear(DenseLayerSize*DenseLayerSize*filter_organisation[4],10)
        self.softmax = nn.Softmax()
        self.learning_rate=0.001
        self.dropout = nn.Dropout(p=drop_out)
        self.save_hyperparameters()

    def forward(self,x):
        out=self.activation_function(self.cnv1(x))
        out=self.mxpool(out)

        out=self.activation_function(self.cnv2(out))
        out=self.mxpool(out)

        out=self.activation_function(self.cnv3(out))
        out=self.mxpool(out)

        out=self.activation_function(self.cnv4(out))
        out=self.mxpool(out)

        out=self.activation_function(self.cnv5(out))
        
        out = self.mxpool(out)
        
        out=self.flat(out)
        
        if(self.batch_normalization=="Yes"):
            out=self.bn(out) 
        
        out = self.dropout(out)

        out = self.activation_function(self.fc(out))
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()


    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

if(args.filter_organisation=="same"):
  filter_organisation1 = [args.filter_size]*5
elif(args.filter_organisation=="half"):   
  filter_organisation1=[args.filter_size,args.filter_size//2,args.filter_size//4,args.filter_size//8,args.filter_size//16]
elif(args.filter_organisation=="double"):
  filter_organisation1=[args.filter_size,args.filter_size*2,args.filter_size*4,args.filter_size*8,args.filter_size*16]


obj = CNNModel(args.activation_function,args.batch_normalization,args.data_augmentation,filter_organisation1, args.drop_out)

trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu", devices=1)

trainer.fit(model=obj,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)

wandb.finish()
