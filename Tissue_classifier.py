import torch.nn.functional as F
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
import os
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pickle
import pandas as pd
import random
import torch
import argparse
#from vit_pytorch import ViT
TARGET = "MSI"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = SwinTransformer(**dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)))
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 9)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.backbone.forward_features(x)  # [batch_size, 768]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Dataset(Dataset):
    def __init__(self, imgs_path, labels, my_transforms):
        self.my_transforms = my_transforms
        self.imgs_path = imgs_path
        self.labels = labels
        self.len = len(self.imgs_path)

    def __getitem__(self, index):
        img = Image.open(self.imgs_path[index])
        return my_transforms(img), self.labels[index]

    def __len__(self):
        return self.len

def cv_pic_list(train_dir,test_dir):
    label_name = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    train_pic_list = []
    train_label_list = []
    test_pic_list = []
    test_label_list = []
    for i in range(len(label_name)):
        train_pic_name = os.listdir(train_dir + "/" + label_name[i])
        for name in train_pic_name:
            train_pic_list.append(train_dir + "/" + label_name[i] + "/" + name)
            train_label_list.append(i)

        test_pic_name = os.listdir(test_dir + "/" + label_name[i])
        for name in test_pic_name:
            test_pic_list.append(test_dir + "/" + label_name[i] + "/" + name)
            test_label_list.append(i)
    return train_pic_list, train_label_list, test_pic_list, test_label_list


def get_args():
    parser = argparse.ArgumentParser(
        description='The script to generate the tiles for Whole Slide Image (WSI).')
    parser.add_argument(
        '-tr', '--train_dir', default="/data/cm/NLP_based_model_code/data/NCT-CRC-HE-100K",
        help='Train_dir Path')
    parser.add_argument('-te', '--test_dir', default="/data/cm/NLP_based_model_code/data/CRC-VAL-HE-7K",
                        help='Test_dir Path')
    parser.add_argument('-lr', '--lr',type = float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('-sp', '--save_path', default="'/data/cm/NLP_based_model_code/pretrained_model/",
                        help='Path to directory in which pre-train model and data will be saved.')

    return parser.parse_args()

if __name__ == ('__main__'):
    args = get_args()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(12345)
    pre_train_model = Net()

    my_transforms = torchvision.transforms.Compose(
        [
            transforms.Resize((224, 224)),  # 重置图片大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.69261899, 0.51298305, 0.7263637], std=[0.13099993, 0.18332116, 0.14441279])])



    train_pic_list, train_label_list, test_pic_list, test_label_list = cv_pic_list(args.train_dir,args.test_dir)


    train_losses = []
    train_counter = []
    test_losses = []
    log_interval=10
    batch_size_train=128
    batch_size_test=128
    model = pre_train_model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = Dataset(train_pic_list, train_label_list, my_transforms)
    test_dataset = Dataset(test_pic_list, test_label_list, my_transforms)

    train_loader = DataLoader(train_dataset, 128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, 128, shuffle=True, num_workers=2)

    for epoch in range(0, 10):
        model.train()
        train_loss = 0
        correct = 0
        y_p = []
        labels = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            targets_to_one_hot = torch.nn.functional.one_hot(target, num_classes=9)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            soft = F.softmax(output, dim=1)[:, 1]
            soft_array = np.array(soft.detach().cpu())
            for i in range(len(soft_array)):
                y_p.append(soft_array[i])
                labels.append(np.array(target.detach().cpu())[i])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
                print(100. * correct / (128 * batch_idx))
        torch.save(model.state_dict(), args.save_path + str(epoch) + '.pth')
        # torch.save(optimizer.state_dict(), '/data/gbw/try_process_WSI/MSI/model/optimizer_SGD' + str(epoch) + '.pth')
        print(100. * correct / len(train_loader.dataset))