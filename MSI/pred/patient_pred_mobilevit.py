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
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import init
from data_utils.patch_dataloader import MSIDataset_256
from efficientnet_pytorch.model import EfficientNet
import torchvision.models as models
#from sequencer_main.models.two_dim_sequencer import sequencer2d_s
#from cmt_pytorch.cmt import cmt_s
from timm.models.mobilevit import mobilevit_s

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn import preprocessing


#%%

def load_patch_model(model_save_path,device):
    """
    This function is used to load trained model
    """
    model_test = torch.load(model_save_path)
    model_test = model_test.to(device)
    return model_test

#def predict_patient_level(patient_path, model,time_total,patch_total, device, MCO = True):
def predict_patient_level(patient_path, model, device, MCO = True):
    """
    This function is used to predict patient level score
    :param patient_path: patient folder contain tiles
    ;param model: loaded model
    ;param device: gpu device
    ;param MCO: if the dataset from MCO dataset

    :return:
    MSI_true: Ground truth of patient
    MSI_pred_avg, MSI_pred_top1, MSI_pred_top10, MSI_pred_top20,MSI_pred_top30, MSI_pred_top50: pred score of patient with different topk
    """
    #torch.cuda.synchronize()
    #start = time.time()
    patch_path_ls = glob.glob(patient_path + '/*.png')
    #n_patch = len(patch_path_ls)
    #patch_total = patch_total + n_patch
    dataset = MSIDataset_256(patch_path_ls,MCO = MCO,train = False)
    test_loader = dataset.get_loader()
 
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    #print(total_num, len(test_loader))
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
            #loss = criterion_val(output, target)
            _, pred = torch.max(output.data, 1)
            pred_prob = F.softmax(output,dim=1)[:,1]
            for p in pred_prob:
                pred_list.append(p.data.item())
            correct += torch.sum(pred == target)
            #print_loss = loss.data.item()
            #test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        #avgloss = test_loss / len(test_loader)

    MSI_true = np.array(val_list).mean()
    MSI_pred_avg = np.array(pred_list).mean()
    #torch.cuda.synchronize()
    #end = time.time()
    #pred_time = end-start
    #time_total = pred_time + time_total
    MSI_pred1 = np.array(pred_list)
    MSI_pred1.sort()
    MSI_pred_top30 = MSI_pred1[-30:].mean()
    MSI_pred_top50 = MSI_pred1[-50:].mean()
    MSI_pred_top20 = MSI_pred1[-20:].mean()
    MSI_pred_top10 = MSI_pred1[-10:].mean()
    MSI_pred_top1 = MSI_pred1[-1:].mean()
    print(MSI_pred_avg,MSI_true)
    
    return MSI_true,MSI_pred_avg, MSI_pred_top1, MSI_pred_top10, MSI_pred_top20,MSI_pred_top30, MSI_pred_top50
    #return MSI_true,MSI_pred_avg,time_total,patch_total


def binary_classification_metric(pred_y_all, true_y_all):
    """
    This function is used to compute different classification metric with numpy array in binary classification mode
    :param pred_y_all: numpy array
    :param true_y_all: numpy array
    :return: acc, auc, kappa_score, confusion_matrix_data, recall_score_data, composite_report
    """

    acc = accuracy_score(true_y_all, pred_y_all.round())
    auc = roc_auc_score(true_y_all, pred_y_all)
    kappa_score = cohen_kappa_score(true_y_all, pred_y_all.round())
    confusion_matrix_data = confusion_matrix(true_y_all, pred_y_all.round())
    return acc, auc, kappa_score, confusion_matrix_data


parser = argparse.ArgumentParser(description="MSI Status Prediction")
parser.add_argument('--batch_size', type=int, default=64, help='has to be 16')
parser.add_argument('--TCGA_folder_path',type=str, default="/data/cm/NLP_based_model_code/data/TCGA_patch_example", help='TCGA tile path')
parser.add_argument('--model_name', type=str, default='MobileViT')
parser.add_argument('--model_save_path', type=str,default = '/data/cm/NLP_based_model_code/outputs/MSI_MobileViT.pth')
parser.add_argument('--output_dir', type=str, default='/data/cm/NLP_based_model_code/outputs')

if __name__ == '__main__':

    args = parser.parse_args([])

    test_folder_path = args.TCGA_folder_path
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    test_auc_list = []
    test_auc_list_top1 = []
    test_auc_list_top10 = []
    test_auc_list_top20 = []
    test_auc_list_top30 = []
    test_auc_list_top50 = []

    model_name = 'MSI_' + args.model_name
    model_save_path = args.model_save_path
    model_test = load_patch_model(model_save_path,device)
    outputdir = args.output_dir

    test_paths = glob.glob(test_folder_path + "/*")
    pred_test = []
    true_test = []
    pred_test_avg = []
    pred_test_top1 = []
    pred_test_top10 = []
    pred_test_top20 = []
    pred_test_top30 = []
    pred_test_top50 = []
    k = 0

    for i in test_paths:
        # predict each patient
        msi_true, MSI_pred_avg, MSI_pred_top1, MSI_pred_top10, MSI_pred_top20,MSI_pred_top30, MSI_pred_top50 = predict_patient_level(i, model_test, device = device,MCO = False)
        pred_test_avg.append(MSI_pred_avg)
        true_test.append(msi_true)
        pred_test_top1.append(MSI_pred_top1)
        pred_test_top10.append(MSI_pred_top10)
        pred_test_top20.append(MSI_pred_top20)
        pred_test_top30.append(MSI_pred_top30)
        pred_test_top50.append(MSI_pred_top50)
        k = k+1
        print(k)
    print(k) 
    pred_df = pd.DataFrame({'test_paths':test_paths,
                            'pred_mean':pred_test_avg,
                            'pred_top50':pred_test_top50,
                            'pred_top30':pred_test_top30,
                            'pred_top20':pred_test_top20,
                            'pred_top10':pred_test_top10,
                            'true_test':true_test})
    pred_df.to_csv(outputdir + '/prediction_results/{}_pred.csv'.format(model_name))
    test_acc, test_auc, test_kappascore, test_cm = binary_classification_metric(np.array(pred_test_avg),np.array(true_test))
    test_acc_top1, test_auc_top1, test_kappascore_top1, test_cm_top1 = binary_classification_metric(np.array(pred_test_top1),np.array(true_test))
    test_acc_top10, test_auc_top10, test_kappascore_top10, test_cm_top10 = binary_classification_metric(np.array(pred_test_top10),np.array(true_test))
    test_acc_top20, test_auc_top20, test_kappascore_top20, test_cm_top20 = binary_classification_metric(np.array(pred_test_top20),np.array(true_test))
    test_acc_top30, test_auc_top30, test_kappascore_top30, test_cm_top30 = binary_classification_metric(np.array(pred_test_top30),np.array(true_test))
    test_acc_top50, test_auc_top50, test_kappascore_top50, test_cm_top50 = binary_classification_metric(np.array(pred_test_top50),np.array(true_test))
    #np.save('/data/cm/MSI_new/pred/Mobilenetv2_pred_{}.npy'.format(fold),np.array(pred_test_avg))
    #np.save('/data/cm/MSI_new/pred/Mobilenetv2_true.npy',np.array(true_test))

    print("k:",k)
    test_auc_list.append(test_auc)
    test_auc_list_top1.append(test_auc_top1)
    test_auc_list_top10.append(test_auc_top10)
    test_auc_list_top20.append(test_auc_top20)
    test_auc_list_top30.append(test_auc_top30)
    test_auc_list_top50.append(test_auc_top50)

    result = pd.DataFrame({'test_auc':test_auc_list,'test_auc_top1':test_auc_list_top1,'test_auc_top10':test_auc_list_top10,'test_auc_top20':test_auc_list_top20,'test_auc_top30':test_auc_list_top30,'test_auc_top50':test_auc_list_top50})
    print(result)
    result.to_csv(outputdir + '/prediction_results/{}_auc.csv'.format(model_name))



# %%
