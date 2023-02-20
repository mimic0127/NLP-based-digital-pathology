# %%
from asyncore import file_dispatcher
import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch.utils.data as Data
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
from torchtoolbox.transform import Cutout
from timm.data.mixup import Mixup
from random import sample
# %%
class MSIDataset(data.Dataset):
    def __init__(self, file_path = [], MCO = True, train = True, batch_size = 64):
        self.imgs_path = file_path
        self.MCO = MCO         
        self.train = train
        self.batch_size = batch_size
        if self.MCO:
            id_msi_df = pd.read_csv('/data/cm/NLP_based_model_code/Ground_Truth/CIMP_label.csv')
            self.id_msi = id_msi_df[['ID','label']]
            self.id_msi.columns = ['ID','CIMP'] 
            # 'MSS','MSI'
        if not self.MCO:
            id_msi_df = pd.read_csv('/data/cm/NLP_based_model_code/Ground_Truth/TCGA_labels.csv')
            self.id_msi = id_msi_df[['ID','HypermethylationCategory']]
            self.id_msi.columns = ['ID','CIMP']
            # 'nonMSIH','MSIH'
        
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = Image.open(img_path)
        pname = img_path[img_path.rindex('/')+1:img_path.rindex('[')]
        plabel = self.id_msi[self.id_msi['ID']==pname]['CIMP'].values
        if plabel == 1 or plabel == 'CIMP-H':
            label = 1
        elif plabel == 0 or plabel == 'Non-CIMP' or plabel == 'CRC CIMP-L':
            label = 0
        else:
            print('error')
        
        img, label = self.img_transform(img, label)

        sample = {'img':img, 'label': label}

        return sample
    
    def __len__(self):
        return len(self.imgs_path)

    def img_transform(self, img, label):
        transform = transforms.Compose(
            [
                transforms.Resize([224,224]),
                #Cutout(),
                #transforms.RandomRotation(degrees=(45, 45)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.69261899, 0.51298305, 0.7263637], std=[0.13099993, 0.18332116, 0.14441279])
                #transforms.Normalize(mean = [0.5677405, 0.120712645, 0.5045699], std = [0.26737085, 0.3536237, 0.24133304])
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.69261899, 0.51298305, 0.7263637], std=[0.13099993, 0.18332116, 0.14441279])
                #transforms.Normalize(mean = [0.5677405, 0.120712645, 0.5045699], std = [0.26737085, 0.3536237, 0.24133304]),
            ]
        )
        if self.train:
            img = transform(img)
        else:
            img = transform_test(img)
        label = torch.tensor(label)

        return img, label

    def get_loader(self):
        if self.train:
            return DataLoader(dataset=self, batch_size=self.batch_size, shuffle=True,num_workers=4,pin_memory=True)
        else:
            return DataLoader(dataset=self, batch_size=self.batch_size, shuffle=False,num_workers=4,pin_memory=True)


class MSIDataset_256(data.Dataset):
    def __init__(self, file_path = [], MCO = True, train = True, batch_size = 64):
        self.imgs_path = file_path
        self.MCO = MCO         
        self.train = train
        self.batch_size = batch_size
        if self.MCO:
            id_msi_df = pd.read_csv('/data/cm/NLP_based_model_code/Ground_Truth/CIMP_label.csv')
            self.id_msi = id_msi_df[['ID','label']]
            self.id_msi.columns = ['ID','CIMP'] 
            # 'MSS','MSI'
        if not self.MCO:
            id_msi_df = pd.read_csv('/data/cm/NLP_based_model_code/Ground_Truth/TCGA_labels.csv')
            self.id_msi = id_msi_df[['ID','HypermethylationCategory']]
            self.id_msi.columns = ['ID','CIMP']
            # 'nonMSIH','MSIH'
        
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = Image.open(img_path)
        pname = img_path[img_path.rindex('/')+1:img_path.rindex('[')]
        plabel = self.id_msi[self.id_msi['ID']==pname]['CIMP'].values
        if plabel == 1 or plabel == 'CIMP-H':
            label = 1
        elif plabel == 0 or plabel == 'Non-CIMP' or plabel == 'CRC CIMP-L':
            label = 0
        else:
            print('error')
        
        img, label = self.img_transform(img, label)

        sample = {'img':img, 'label': label}

        return sample
    
    def __len__(self):
        return len(self.imgs_path)


    def img_transform(self, img, label):
        transform = transforms.Compose(
            [
                transforms.Resize([256,256]),
                #Cutout(),
                #transforms.RandomRotation(degrees=(45, 45)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.69261899, 0.51298305, 0.7263637], std=[0.13099993, 0.18332116, 0.14441279])
                #transforms.Normalize(mean = [0.5677405, 0.120712645, 0.5045699], std = [0.26737085, 0.3536237, 0.24133304])
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize([256,256]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.69261899, 0.51298305, 0.7263637], std=[0.13099993, 0.18332116, 0.14441279])
                #transforms.Normalize(mean = [0.5677405, 0.120712645, 0.5045699], std = [0.26737085, 0.3536237, 0.24133304]),
            ]
        )
        if self.train:
            img = transform(img)
        else:
            img = transform_test(img)
        label = torch.tensor(label)

        return img, label

    def get_loader(self):
        if self.train:
            return DataLoader(dataset=self, batch_size=self.batch_size, shuffle=True,num_workers=4,pin_memory=True)
        else:
            return DataLoader(dataset=self, batch_size=self.batch_size, shuffle=False,num_workers=4,pin_memory=True)


if __name__ == '__main__':
    # %%
    print('done')
