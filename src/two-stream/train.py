import numpy as np
import torch
from resnet18 import ResBlock, ResNet
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
from torch.nn import functional as F
import copy
from GetDataSet import MyDataSet, GetDataList
import torch.utils.data as Data


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_save_path', type=str, help='the path where the model is saved')
    argparser.add_argument('--anl_or_dfs', type=int, help='train content-consistency network or homology network, 0 for former, 1 for latter')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=50)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--learning_rate', type=int, help='learning rate', default=1e-4)
    argparser.add_argument('--is_reload_model', action='store_true', help='reload model or not')
    args = argparser.parse_args()

    nonlinear_or_dfs = args.anl_or_dfs
    batch_size = args.batch_size
    total_epoch = args.epoch
    learning_rate = args.learning_rate
    model_save_path = args.model_save_path

    # get train dataset
    attack_type = ['normal', 'L1H1'] # choose the data of different attack method
    train_dataset, test_dataset, train_dataloader, test_dataloader = GetDataList(cfg.root_path_train, attack_type, batch_size)

    device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
    print(device)

    model = ResNet(ResBlock).to(device)
    opt = torch.optim.Adam([
            {'params': model.parameters(), 'lr': learning_rate, 'weight_decay':1e-4}, 
            ])

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.25)


    epoch_start = 0
    if args.is_reload_model:
        print("reloading model...")
        ckpt = torch.load(model_save_path)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        epoch_start = ckpt['epoch']
    print(f"start from epoch {epoch_start+1}")
    loss_func = nn.BCELoss()
    

    for epoch in range(epoch_start+1, total_epoch+1):
     
        #train
        loss_total= 0
        right_num = 0

        model.train()
    
        for step, (voice, ultra, label) in enumerate(train_dataloader):   
            voice = voice.to(device)
            ultra = ultra.to(device)
            label = label.to(device)
  
            input = torch.cat((voice, ultra), dim=1)
            out = model(input)
            loss = loss_func(out, label)
            loss_total += loss.item() 
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            pred = out
            pred[pred>0.8] = 1
            pred[pred<=0.8] = 0
            right_num += sum(pred==label).item()
            print("Step: {}/{} | loss: {:.4f}".format(step+1, len(train_dataloader), loss), end='\r')
        
        print("Epoch: {}/{} | train loss: {:.4f} | acc: {:.4f}".format(epoch, total_epoch, loss_total/len(train_dataloader), right_num/len(train_dataset)))
       
        #eval

        loss_total = 0
        right_num = 0

        FP_num = 0
        FN_num = 0
        TP_num = 0
        TN_num = 0
        T_num = 0
        N_num = 0

        
        count = 0
        model.eval()
        with torch.no_grad():
            for step, (voice, ultra, label) in enumerate(test_dataloader):   
                
                voice = voice.to(device)
                ultra = ultra.to(device)
                label = label.to(device)
    

                input = torch.cat((voice, ultra), dim=1)
                out = model(input)
        

                loss = loss_func(out, label)
                loss_total += loss.item()

                pred = out
                pred[pred>0.8] = 1
                pred[pred<=0.8] = 0
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
            
            print("Epoch: {}/{} | test loss: {:.4f} | acc: {:.4f}".format(epoch, total_epoch, loss_total/len(test_dataloader), right_num/len(test_dataset)))
       
            print("FP Rate:%.4f, FN Rate:%.4f, Precision:%.4f, Recall:%.4f"% (FP_num/(N_num+1e-6), FN_num/(T_num+1e-6), TP_num/(TP_num+FP_num+1e-6), TP_num/(TP_num+FN_num+1e-6)))
        scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),  
            'scheduler_state_dict': scheduler.state_dict(),
            }, model_save_path)
            






    
