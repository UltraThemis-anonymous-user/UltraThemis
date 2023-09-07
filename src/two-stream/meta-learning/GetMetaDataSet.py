import numpy as np
from pandas import read_pickle
from sklearn import model_selection
import config as cfg
import torch.utils.data as Data
from torch.utils.data import Dataset
import re
import os
import random
import matplotlib.pyplot as plt
import matplotlib
import torch

class MetaDataSet(Dataset):
    """
    inputs:
        data_path_list: the path of data with DFS and data without DFS
        mode: train or test
        batch_num
        person_num: the number of new user(default 1) + the number of person used for training
        k_support: k shot for support set
        k_query: k shot for query set
        new_user_list: the list of new user which is used for testing
    """
    def __init__(self, data_path_list, mode, batch_num, person_num, k_support, k_query, new_user_list=[]) -> None:
        super().__init__() 
        assert len(data_path_list) == 2, 'Both data with dfs and without dfs need to be provided'
        self.batch_num = batch_num
        self.person_num = person_num
        self.k_support = k_support
        self.k_query = k_query
        self.supportsz = (self.person_num-1)*self.k_support*2
        self.querysz = (self.person_num-1)*self.k_query*2
        self.enroll_name_list = []
        self.other_name_list = []
        path = os.path.join(data_path_list[0], "voice")
        file_list = os.listdir(path)
        file_list.sort()
        for name in file_list:
            if mode=="train":
                self.enroll_name_list += [name]
                self.other_name_list += [name]
            if mode=="test":
                if name in new_user_list:
                    self.enroll_name_list += [name]
                else:
                    self.other_name_list += [name]
        print("enroll person: ", self.enroll_name_list)
        print("other person: ", self.other_name_list)
        self.clean_voice_list = {}
        self.clean_ultra_list = {}
        self.nodoppler_voice_list = {}
        self.nodoppler_ultra_list = {}
    
        #get voice segments
        path = os.path.join(data_path_list[0], "voice/")
        folder_list = os.listdir(path)
        folder_list.sort()
        for folder in folder_list:
            voice_file_list = []
            if folder not in self.enroll_name_list and folder not in self.other_name_list:
                continue
            files = os.listdir(path+folder)
            files.sort(key = lambda x: (int(re.split('-|.npy', x)[0])))
            voice_file_list = [path+folder + "/"+file for file in files]
            self.clean_voice_list[folder] = voice_file_list
        

        #get ultra segments
        path = os.path.join(data_path_list[0], "ultra/")
        folder_list = os.listdir(path)
        folder_list.sort()
        for folder in folder_list:
            ultra_file_list = []
            if folder not in self.enroll_name_list and folder not in self.other_name_list:
                continue
            files = os.listdir(path+folder)
            files.sort(key = lambda x: (int(re.split('-|.npy', x)[0])))
            ultra_file_list = [path+folder + "/"+file for file in files]
            self.clean_ultra_list[folder] = ultra_file_list

        #get nodoppler voice segments
        path = os.path.join(data_path_list[1], "voice/")
        folder_list = os.listdir(path)
        folder_list.sort()
        for folder in folder_list:
            voice_file_list = []
            if folder not in self.enroll_name_list and folder not in self.other_name_list:
                continue
            files = os.listdir(path+folder)
            files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))
            voice_file_list = [path+folder + "/"+file for file in files]
            self.nodoppler_voice_list[folder] = voice_file_list
        

        #get nodoppler ultra segments
        path = os.path.join(data_path_list[1], "ultra/")
        folder_list = os.listdir(path)
        folder_list.sort()
        for folder in folder_list:
            ultra_file_list = []
            if folder not in self.enroll_name_list and folder not in self.other_name_list:
                continue
            files = os.listdir(path+folder)
            files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))
            ultra_file_list = [path+folder + "/"+file for file in files]
            self.nodoppler_ultra_list[folder] = ultra_file_list

        
        self.create_batch()

    def create_batch(self):
        self.support_ultra_batch = []
        self.support_voice_batch = []
        self.support_y_batch = []
        self.query_ultra_batch = []
        self.query_voice_batch = []
        self.query_y_batch = []

        for b in range(self.batch_num):
            name_list = []
            #select enroll person
            enroll_idx = np.random.choice(len(self.enroll_name_list), size=1, replace=False)[0]
            enroll_name = self.enroll_name_list[enroll_idx]
            #select other person
            for i in range(self.person_num-2):
                idx = np.random.choice(len(self.other_name_list), size=1, replace=False)[0]
                while idx==enroll_idx:
                    idx = np.random.choice(len(self.other_name_list), size=1, replace=False)[0]
                name_list += [self.other_name_list[idx]]
      
            support_x_ultra, support_x_voice, support_y = [], [], []
            query_x_ultra, query_x_voice, query_y = [], [], []
            # print(name_list)


            #select positive samples
            selected_ultra_idx = np.random.choice(len(self.clean_ultra_list[enroll_name]), self.supportsz//2+self.querysz//2, replace=True)
            selected_voice_idx = np.random.choice(len(self.clean_voice_list[enroll_name]), self.supportsz//2+self.querysz//2, replace=True)
            selected_ultra = np.array(self.clean_ultra_list[enroll_name])[selected_ultra_idx].tolist()
            selected_voice = np.array(self.clean_voice_list[enroll_name])[selected_voice_idx].tolist()
            support_x_ultra += selected_ultra[:self.supportsz//2]
            support_x_voice += selected_voice[:self.supportsz//2]
            query_x_ultra += selected_ultra[self.supportsz//2:]
            query_x_voice += selected_voice[self.supportsz//2:]
            support_y += [1]*(self.supportsz//2)
            query_y += [1]*(self.querysz//2)

            #select nodoppler negative samples
            selected_ultra_idx = np.random.choice(len(self.nodoppler_ultra_list[enroll_name]), self.k_support+self.k_query, replace=True)
            selected_voice_idx = np.random.choice(len(self.nodoppler_voice_list[enroll_name]), self.k_support+self.k_query, replace=True)
            selected_ultra = np.array(self.nodoppler_ultra_list[enroll_name])[selected_ultra_idx].tolist()
            selected_voice = np.array(self.nodoppler_voice_list[enroll_name])[selected_voice_idx].tolist()
            support_x_ultra += selected_ultra[:self.k_support]
            support_x_voice += selected_voice[:self.k_support]
            query_x_ultra += selected_ultra[self.k_support:]
            query_x_voice += selected_voice[self.k_support:]
            support_y += [0]*(self.k_support)
            query_y += [0]*(self.k_query)

            
            #select negative samples
            for name in name_list:
                person = [enroll_name, name]
                selected_ultra = []
                selected_voice = []
                for i in range(self.k_support+self.k_query):
                    #select one person's ultra and the other's voice randomly
                    u_idx = np.random.choice(2, 2, False)
                    selected_ultra_idx = np.random.choice(len(self.clean_ultra_list[person[u_idx[0]]]), 1, replace=True)[0]
                    selected_voice_idx = np.random.choice(len(self.clean_voice_list[person[u_idx[1]]]), 1, replace=True)[0]
                    selected_ultra += [self.clean_ultra_list[person[u_idx[0]]][selected_ultra_idx]]
                    selected_voice += [self.clean_voice_list[person[u_idx[1]]][selected_voice_idx]]
                
                #split support set and query set
                support_x_ultra += selected_ultra[:self.k_support]
                support_x_voice += selected_voice[:self.k_support]
                query_x_ultra += selected_ultra[self.k_support:]
                query_x_voice += selected_voice[self.k_support:]

                support_y += [0]*self.k_support
                query_y += [0]*self.k_query

            length = len(support_x_ultra)
            idx = random.sample(range(length), length)
            support_x_ultra = np.array(support_x_ultra)[idx].tolist()
            support_x_voice = np.array(support_x_voice)[idx].tolist()
            support_y = np.array(support_y)[idx].tolist()

            length = len(query_x_ultra)
            idx = random.sample(range(length), length)
            query_x_ultra = np.array(query_x_ultra)[idx].tolist()
            query_x_voice = np.array(query_x_voice)[idx].tolist()
            query_y = np.array(query_y)[idx].tolist()

            self.support_ultra_batch.append(support_x_ultra)
            self.support_voice_batch.append(support_x_voice)
            self.support_y_batch.append(support_y)

            self.query_ultra_batch.append(query_x_ultra)
            self.query_voice_batch.append(query_x_voice)
            self.query_y_batch.append(query_y)


        pass

    def __getitem__(self, index):
        support_ultra = self.support_ultra_batch[index]
        support_voice = self.support_voice_batch[index]
        support_y = self.support_y_batch[index]
        query_ultra = self.query_ultra_batch[index]
        query_voice = self.query_voice_batch[index]
        query_y = self.query_y_batch[index]

        support_ultra = np.array([np.load(path) for path in support_ultra])
        support_voice = np.array([np.load(path) for path in support_voice])
        support_y = np.array(support_y)
        support_y = support_y.reshape((-1, 1))
        query_ultra = np.array([np.load(path) for path in query_ultra])
        query_voice = np.array([np.load(path) for path in query_voice])
        query_y = np.array(query_y)
        query_y = query_y.reshape((-1, 1))
         

        support_ultra = torch.tensor(support_ultra, dtype=torch.float32, requires_grad=True)
        support_voice = torch.tensor(support_voice, dtype=torch.float32, requires_grad=True)
        support_y = torch.tensor(support_y, dtype=torch.float32, requires_grad=True)
     
        query_ultra = torch.tensor(query_ultra, dtype=torch.float32, requires_grad=True)
        query_voice = torch.tensor(query_voice, dtype=torch.float32, requires_grad=True)
        query_y = torch.tensor(query_y, dtype=torch.float32, requires_grad=True)
 

        return support_ultra, support_voice, support_y, query_ultra, query_voice, query_y

    def __len__(self):
        return self.batch_num
    