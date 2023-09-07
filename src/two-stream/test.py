import enum
from venv import create
import numpy as np
import torch
from resnet18 import ResNet, ResBlock
import config as cfg
import os
import  argparse 
import re
import random
import math
import torch.nn as nn
import sys
from torch.autograd.variable import Variable
from numpy.core.fromnumeric import argmax
import matplotlib.pyplot as plt
from torch.nn import functional as F
from GetDataSet import MyDataSet, GetDataList
import torch.utils.data as Data
import shutil



if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_save_path1', type=str, help='the path where the model of content consistency network is saved')
    argparser.add_argument('--model_save_path2', type=str, help='the path where the model of homology network is saved')
    argparser.add_argument('--model_save_path3', type=str, help='the path where the model of meta learning is saved')
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--threshold1', type=float, help='threshold for content consistency network', default=0.8)
    argparser.add_argument('--threshold2', type=float, help='threshold for homology network', default=0.4)
    argparser.add_argument('--threshold3', type=float, help='threshold for meta learning model', default=0.4)
    args = argparser.parse_args()

    batch_size = args.batch_size
    model_save_path1 = args.model_save_path1
    model_save_path2 = args.model_save_path2
    model_save_path3 = args.model_save_path3
    threshold1 = args.threshold1
    threshold2 = args.threshold2
    threshold3 = args.threshold3
    # get test dataset
    attack_type = ['normal', 'L1H1'] # choose the data of different attack method
    train_dataset, test_dataset, train_dataloader, test_dataloader = GetDataList(cfg.root_path_test, attack_type, batch_size)

    device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
    print(device)
    model1 = ResNet(ResBlock).to(device)
    model2 = ResNet(ResBlock).to(device)
    model3 = ResNet(ResBlock).to(device)

    ckpt = torch.load(model_save_path1)
    model1.load_state_dict(ckpt['model_state_dict'])
    ckpt = torch.load(model_save_path2)
    model2.load_state_dict(ckpt['model_state_dict'])
    ckpt = torch.load(model_save_path3)
    model3.load_state_dict(ckpt['model_state_dict'])

    loss_func = nn.BCELoss()
 
    loss_total = 0
  
    right_num = 0
    FP_num = 0
    FN_num = 0
    TP_num = 0
    TN_num = 0
    T_num = 0
    N_num = 0

    model1.eval()
    model2.eval()
    model3.eval()
    with torch.no_grad():
        for step, (voice, ultra, label) in enumerate(test_dataloader):   
            
            voice = voice.to(device)
            ultra = ultra.to(device)
            label = label.to(device)


            input = torch.cat((voice, ultra), dim=1)
            pred1 = model1(input)
            pred2 = model2(input)


            loss_nonlinear = loss_func(pred1, label)
            loss_doppler = loss_func(pred2, label)
            loss_total += loss_nonlinear.item() + loss_doppler.item()
            loss = loss_nonlinear.item() + loss_doppler.item() 

            
            pred1[pred1>threshold1] = 1
            pred1[pred1<=threshold1] = 0
            pred2[pred2>threshold2] = 1
            pred2[pred2<=threshold2] = 0

            index = pred1[:, 0]==1
            voice2 = voice[index,]
            ultra2 = ultra[index,]
            label2 = label[index,]
            input = torch.cat((voice2, ultra2), dim=1)
            if len(input)!=0:
                pred3 = model3(input)
                pred3[pred3>threshold3] = 1
                pred3[pred3<=threshold3] = 0
                j=0
                for i, v in enumerate(index):
                    if v==True:
                        pred2[i] = pred2[i] or pred3[j]
                        j+=1
            pred = pred1*pred2


            right_num += sum(pred==label).item()

            for i in range(len(pred)):
                if label[i]==0:
                    N_num += 1
                if pred[i]==1 and label[i]==0:
                    FP_num += 1
                if pred[i]==0 and label[i]==0:
                    TN_num += 1

                if label[i]==1:
                    T_num += 1
                if pred[i]==0 and label[i]==1:
                    FN_num += 1
                if pred[i]==1 and label[i]==1:
                    TP_num += 1
        
            
            print("Step: {}/{} | loss: {:.4f}".format(step+1, len(test_dataloader), loss), end='\r')
        
        print("test loss: {:.4f} | acc: {:.4f}".format(loss_total/len(test_dataloader), right_num/len(test_dataset)))

        print("FP Rate:%.4f, FN Rate:%.4f, Precision:%.4f, Recall:%.4f"% (FP_num/(N_num+1e-6), FN_num/(T_num+1e-6), TP_num/(TP_num+FP_num+1e-6), TP_num/(TP_num+FN_num+1e-6)))

