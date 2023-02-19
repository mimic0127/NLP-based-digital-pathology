# %%
import argparse
import time
import numpy as np
import pandas as pd
import os
import torch
import glob
import gc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import random
from early_stopping import EarlyStopping
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from timm.data.mixup import Mixup
from sequencer_main.models.two_dim_sequencer import sequencer2d_s
from efficientnet_pytorch.model import EfficientNet
#from apex import amp
# %%
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import init
from data_utils.patch_dataloader import MSIDataset_256
from timm.models.mobilevit import mobilevit_s
from cmt_pytorch.cmt import cmt_s,cmt_b

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark=True


# %%
parser = argparse.ArgumentParser(description="MSI_Score_model")
parser.add_argument("--nepochs", type=int, default=1, help="The maxium number of epochs to train")
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=32, help='default: 64')
parser.add_argument('--TCGA_folder_path',type=str, default="/data/cm/NLP_based_model_code/data/TCGA_patch_example", help='patch path for test')
parser.add_argument('--MCO_MSS_path',type=str, default="/data/cm/NLP_based_model_code/data/MCO_MSS_example",help='patch path for train with status-0')
parser.add_argument('--MCO_MSI_path',type=str, default="/data/cm/NLP_based_model_code/data/MCO_MSI_example",help='patch path for train with status-1')
parser.add_argument('--model_name',type=str, default='MobileViT', help='model name')
parser.add_argument('--output_dir',type=str, default='/data/cm/NLP_based_model_code/outputs',help='the path dir to save result and model')
parser.add_argument('--model_save_path', type=str, default=None, help='the path to save model')


def binary_classification_metric(pred_y_all, true_y_all):
    """
    This function is used to compute different classification metric with numpy array in binary classification mode
    :param pred_y_all: numpy array
    :param true_y_all: numpy array
    :return: acc, auc, kappa_score, confusion_matrix_data
    """

    acc = accuracy_score(true_y_all, pred_y_all.round())
    auc = roc_auc_score(true_y_all, pred_y_all)
    kappa_score = cohen_kappa_score(true_y_all, pred_y_all.round())
    confusion_matrix_data = confusion_matrix(true_y_all, pred_y_all.round())

    return acc, auc, kappa_score, confusion_matrix_data


def train_epoch(model, device, train_loader, optimizer, epoch, criterion):
    """
    This function is used to train model in one epoch
    :param model: the model to train
    :param device: gpu device
    :param train_loader: data_loader for train
    :param optimizer: optimizer
    :param epoch: the epoch now training
    :param criterion: loss function
    """

    torch.cuda.synchronize()
    start = time.time()
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, sample in enumerate(train_loader):
        data = sample['img']
        target = sample['label']
        data, target = data.to(device), target.to(device)
        #samples, targets = mixup_fn(data, target)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    torch.cuda.synchronize()
    end = time.time()
    print("Time: {}".format(end-start))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))


def prediction(model, device, test_loader,criterion_val,epoch,model_save_path,test_flag = False, model_early_stopping=None):
    """
    This function is used for prediction in validation and test 
    :param model: the trained model
    :param device: gpu device
    :param test_loader: dataloader for validate and test
    :param criterion_val: loss function for validation
    :param epoch: the epoch now training
    :param model_save_path: path to save model
    :param test_flag: if this function is for test
    ;param model_early_stopping: early stopping for model training

    :return:
    val_list: list of ground truth of tiles
    pred_list: list of pred scores of tiles
    model_es: early stopping flag
    """

    global AUC
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    val_list = []
    pred_list = []
    with torch.no_grad():
        for sample in test_loader:
            data = sample['img']
            target = sample['label']
            for t in target:
                val_list.append(t.data.item())
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion_val(output, target)
            _, pred = torch.max(output.data, 1)
            pred_prob = F.softmax(output,dim=1)[:,1]
            for p in pred_prob:
                pred_list.append(p.data.item())
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        acc, val_auc, kappa_score, confusion_matrix_data = binary_classification_metric(np.array(pred_list), np.array(val_list))
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), AUC: {:.6f}\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc,val_auc))
        model_es = False
        if test_flag != True:
            model_early_stopping(avgloss, model)
            # save model that achieve the highest AUC on validation dataset 
            if val_auc > AUC:
                torch.save(model, model_save_path)
                AUC = val_auc
            # if avgloss < AUC:
            #     torch.save(model, model_save_path)
            #     AUC = avgloss
            if model_early_stopping.early_stop:
                model_es = True
    return val_list, pred_list, model_es


def train(model_name, train_path_list, val_path_list, test_path_list, model_save_path, num_epochs, lr, device):
    """
    This function is used to train the model and test the model
    :param model name: the name model to choose
    :param train_path_list: path list of tiles for training
    :param val_path_list: path list of tiles for validation
    :param test_path_list: path list of tiles for test
    :param model_save_path: path to save model
    :param num_epoch: the number of epochs
    :param lr: learning rate
    :param device: gpu device
    """
    train_data = MSIDataset_256(train_path_list, train=True,MCO=True)
    train_loader = train_data.get_loader()

    val_data = MSIDataset_256(val_path_list, train=False,MCO=True)
    val_loader = val_data.get_loader()

    test_data = MSIDataset_256(test_path_list, train=False,MCO=False)
    test_loader = test_data.get_loader()
 
    # model definition
    if model_name == 'MobileViT':
        model = mobilevit_s()  # 定义模型，并设置预训练
        model.load_state_dict(torch.load('/data/cm/NLP_based_model_code/pretrained_model/mobilevit_s-38a5a959.pth')) 
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, 2)
        print('model load success')
        model = model.to(device)
    else:
        print('model_name error!')

    print("load model success!!!!!!!!")
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([0.15,1-0.15]).float().to(device))
    criterion_val = torch.nn.CrossEntropyLoss(weight = torch.tensor([0.15,1-0.15]).float().to(device))  # 验证用的loss
    model_early_stopping = EarlyStopping(model_path=model_save_path, patience=8, verbose=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)      

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)  # 使用余弦退火算法调整学习率

    is_set_lr = False
    for epoch in range(1, num_epochs + 1):
        # train
        train_epoch(model, device, train_loader, optimizer, epoch,criterion)
        if epoch < 600:
            cosine_schedule.step()
        else:
            if is_set_lr:
                continue
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-6
                is_set_lr = True

        # validation
        val_list, pred_list, earlystopping = prediction(model, device, val_loader,criterion_val,epoch,model_save_path,model_early_stopping = model_early_stopping)
        if earlystopping:
            print("Early stopping")
            break
    # test
    print('test_data_predicting...')
    model_test = torch.load(model_save_path)
    model_test = model_test.to(device)
    test_list, pred_test_list,_ = prediction(model_test, device, test_loader, criterion_val, epoch, model_save_path, test_flag=True)
    #print(classification_report(val_list, pred_list))


# %%
if __name__ == '__main__':
    # %%
    args = parser.parse_args([])

    model_name = args.model_name
    batch_size = args.batch_size
    num_epochs = args.nepochs
    lr = args.lr

    test_folder_path = args.TCGA_folder_path
    MCO_MSS_path = args.MCO_MSS_path
    MCO_MSI_path = args.MCO_MSI_path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.model_save_path == None:
        model_save_path = args.output_dir + '/MSI_' + model_name + '.pth'
    else:
        model_save_path = args.model_save_path

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=5, random_state=778, shuffle=True)
    fold = 0

    classes = 2
    # %%
    # depend on patient  
    ID_path_0 = glob.glob(MCO_MSS_path+'/*')
    ID_path_1 = glob.glob(MCO_MSI_path + '/*')
    index_0 = range(len(ID_path_0))
    index_1 = range(len(ID_path_1))
    test_feature_paths = glob.glob(test_folder_path + "/*/*.png")

    # training parameter setting
    print("model_name:", model_name)
    print("batch_size:", batch_size)
    print("num_epochs:", num_epochs)
    print("learning_rate:", lr)

    for train_index, val_index in kf.split(index_0):
        if fold != 0:
            fold += 1
            continue
        flag = 0
        for train_index_1, val_index_1 in kf.split(index_1):
            if flag == fold:
                break
            else:
                flag += 1
        
        # process data (split datasets)
        test_path = test_feature_paths
        train_path_00 = [ID_path_0[j] for j in train_index]
        train_path_0 = []
        for i in train_path_00:
            train_path_0 = train_path_0 + glob.glob(i + '/*.png')
        train_path_11 = [ID_path_1[j] for j in train_index_1]
        train_path_1 = []
        for i in train_path_11:
            train_path_1 = train_path_1 + glob.glob(i + '/*.png')
        train_path = train_path_0 + train_path_1
        random.seed(34)
        random.shuffle(train_path)
        val_path_00 = [ID_path_0[j] for j in val_index]
        val_path_0 = []
        for i in val_path_00:
            val_path_0 = val_path_0 + glob.glob(i + '/*.png')
        val_path_11 = [ID_path_1[j] for j in val_index_1]
        val_path_1 = []
        for i in val_path_11:
            val_path_1 = val_path_1 + glob.glob(i + '/*.png')
        val_path = val_path_0 + val_path_1
        random.seed(34)
        random.shuffle(val_path)

        # training
        AUC = 0
        print("Now training fold:{}".format(fold))
        train(model_name, train_path, val_path, test_path, model_save_path,  num_epochs=num_epochs,
                                              lr=lr, device=device)

        fold += 1