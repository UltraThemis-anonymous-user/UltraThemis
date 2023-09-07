from tkinter.tix import Tree
import numpy as np
import os
import re
from pandas import test
from sklearn import model_selection
import config as cfg
import torch.utils.data as Data
from torch.utils.data import Dataset
import torch
from torch.autograd.variable import Variable

class MyDataSet(Dataset):
    def __init__(self, voice_list, ultra_list, labels):
  
        self.voice_list = voice_list
        self.ultra_list = ultra_list
        self.labels = labels
    def __getitem__(self, index):

        voice_path = self.voice_list[index]
        ultra_path = self.ultra_list[index]
        label = self.labels[index]
        voice = np.load(voice_path)
        ultra = np.load(ultra_path)

        voice = torch.tensor(voice, dtype=torch.float32, requires_grad=True)
        ultra = torch.tensor(ultra, dtype=torch.float32, requires_grad=True)
        label = torch.tensor(label, dtype=torch.float32, requires_grad=True)
        label = label.reshape((1))
        return voice, ultra, label
    def __len__(self):
        return len(self.voice_list)


def GetAndSplitDataFile(root_path, attack_type_list, save_path="", train_size=0.8):

    if not isinstance(attack_type_list, list):
        attack_type_list = [attack_type_list]
        
    train_voice_list = []
    train_ultra_list = []
    train_labels = []
    test_voice_list = []
    test_ultra_list = []
    test_labels = []

    
    for attack_type in attack_type_list:
        #get voice segments
        path = os.path.join(root_path, attack_type, "voice/")
        folder_list = os.listdir(path)
        folder_list.sort()
        voice_file_list = []
        for folder in folder_list:
            files = os.listdir(path+folder)
            files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))
            voice_file_list = voice_file_list+[path+folder + "/"+file for file in files]

        #get ultra segments
        path = os.path.join(root_path, attack_type, "ultra/")
        folder_list = os.listdir(path)
        folder_list.sort()
        ultra_file_list = []
        for folder in folder_list:
            files = os.listdir(path+folder)
            files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))
            ultra_file_list = ultra_file_list+[path+folder + "/"+file for file in files]
    
        label_tmp = []

        # get label
        if attack_type in cfg.positive_data_type:
            label_tmp += [1]*len(voice_file_list)
        else:
            label_tmp += [0]*len(voice_file_list)
        
        #split
        if train_size==0:
            train_voice=[]
            test_voice=voice_file_list
            train_ultra=[]
            test_ultra=ultra_file_list
            train_label=[]
            test_label=label_tmp
        else:
            train_voice, test_voice, train_ultra, test_ultra, train_label, test_label = model_selection.train_test_split(voice_file_list, ultra_file_list, label_tmp, train_size=train_size)
        train_voice_list += train_voice
        test_voice_list += test_voice
        train_ultra_list += train_ultra
        test_ultra_list += test_ultra
        train_labels += train_label
        test_labels += test_label
    # save paired data
    if save_path!="":
        np.save(save_path+"/train_voice_list.npy", train_voice_list)
        np.save(save_path+"/test_voice_list.npy", test_voice_list)
        np.save(save_path+"/train_ultra_list.npy", train_ultra_list)
        np.save(save_path+"/test_ultra_list.npy", test_ultra_list)
        np.save(save_path+"/train_labels.npy", train_labels)
        np.save(save_path+"/test_labels.npy", test_labels)

    print("train: ", np.shape(train_voice_list))
    print("test: ", np.shape(test_voice_list))
    return train_voice_list, test_voice_list, train_ultra_list, test_ultra_list, train_labels, test_labels


def GetDataList(data_path, attack_type, batch_size):
    if os.path.exists(data_path+"/train_voice_list.npy"):
        train_voice_list = np.load(data_path+"/train_voice_list.npy", )
        test_voice_list = np.load(data_path+"/test_voice_list.npy", )
        train_ultra_list = np.load(data_path+"/train_ultra_list.npy", )
        test_ultra_list = np.load(data_path+"/test_ultra_list.npy", )
        train_labels = np.load(data_path+"/train_labels.npy", )
        test_labels = np.load(data_path+"/test_labels.npy", )
    else:  
        train_voice_list, test_voice_list, train_ultra_list, test_ultra_list, train_labels, test_labels = \
                    GetAndSplitDataFile(cfg.root_path, attack_type, save_path=data_path)
        
    if len(train_voice_list)!=0:
        train_dataset = MyDataSet(train_voice_list, train_ultra_list, train_labels) 
        train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = None

    if len(test_voice_list)!=0:
        test_dataset = MyDataSet(test_voice_list, test_ultra_list, test_labels)
        test_dataloader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_dataloader = None
    return train_dataset, test_dataset, train_dataloader, test_dataloader
