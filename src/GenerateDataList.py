from json import load
import numpy as np
import config as cfg
import os 
import re
import random
import math
from numpy.core.fromnumeric import argmax
import matplotlib.pyplot as plt
import hyperparameters as hp
import copy
from utils.utils import create_file


def GenerateDataList():
    interval = hp.interval
    #get original ultrasound file
    path = cfg.ultra_doppler_path
    is_begin = 1
    folder_list = os.listdir(path)
    folder_list.sort()
    ultra_doppler_list = []
    for folder in folder_list:
        files = os.listdir(path+folder+"/"+str(interval))
        files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1])))

        ultra_doppler_list += [path+folder+"/"+str(interval) + "/"+file for file in files]

    ultra_doppler_list = np.array(ultra_doppler_list)
    ultra_doppler_list = ultra_doppler_list.repeat(hp.noisy_audio_num, axis=0)


    #get clean speech file
    folder_list = os.listdir(cfg.speech_doppler_path)
    folder_list.sort()
    speech_doppler_list = []
    for folder in folder_list:
        files = os.listdir(cfg.speech_doppler_path+folder+"/"+str(interval))
        files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))

        speech_doppler_list += [path+folder+"/"+str(interval) + "/"+file for file in files]
    speech_doppler_list = np.array(speech_doppler_list)


    #get noisy audio file
    folder_list = os.listdir(cfg.noisy_speech_doppler_path)
    folder_list.sort()
    noisy_speech_doppler_list = []
    for folder in folder_list:
        # if folder!="zlf":
        #     continue
        files = os.listdir(cfg.noisy_amplitude+folder+"/"+str(interval))
        files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))
        noisy_speech_doppler_list += [path+folder+"/"+str(interval) + "/"+file for file in files]
    
    noisy_speech_doppler_list = np.array(noisy_speech_doppler_list)

    


    #get attack ultrasound file
    folder_list = os.listdir(cfg.attack_ultra_doppler_path)
    folder_list.sort()
    attack_ultra_doppler_list = []
    for folder in folder_list:

        files = os.listdir(cfg.dislocate_ultra_doppler+folder+"/"+str(interval))
        files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))
        attack_ultra_doppler_list += [path+folder+"/"+str(interval) + "/"+file for file in files]
        
    attack_ultra_doppler_list = np.array(attack_ultra_doppler_list)

    #get attack speech file
    folder_list = os.listdir(cfg.attack_speech_doppler_path)
    folder_list.sort()
    attack_speech_doppler_list = []
    for folder in folder_list:

        files = os.listdir(cfg.dislocate_clean_doppler+folder+"/"+str(interval))
        files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))
        attack_speech_doppler_list += [path+folder+"/"+str(interval) + "/"+file for file in files]
    attack_speech_doppler_list = np.array(attack_speech_doppler_list)

    #get original speech file
    folder_list = os.listdir(cfg.original_speech_doppler_path)
    folder_list.sort()
    original_speech_doppler_list = []
    for folder in folder_list:

        files = os.listdir(cfg.dislocate_clean_doppler+folder+"/"+str(interval))
        files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))
        original_speech_doppler_list += [path+folder+"/"+str(interval) + "/"+file for file in files]
    original_speech_doppler_list = np.array(original_speech_doppler_list)



    #get location label
    path = cfg.attack_label_path
    folder_list = os.listdir(path)
    folder_list.sort()
    rg_label = []
    for folder in folder_list:
        files = os.listdir(path+folder+"/"+str(interval))
        files.sort(key = lambda x: (int(re.split('-|.npy', x)[0]), int(re.split('-|.npy', x)[1]), int(re.split('-|.npy', x)[2])))
        rg_label += [path+folder+"/"+str(interval) + "/"+file for file in files]
    rg_label = np.array(rg_label)
    rg_ultra = copy.deepcopy(attack_ultra_doppler_list)
    rg_audio = copy.deepcopy(attack_speech_doppler_list)

    cl_label = [0]*len(attack_speech_doppler_list) + [1]*len(original_speech_doppler_list)
    cl_label = np.array(cl_label)
    cl_ultra = np.append(attack_ultra_doppler_list, attack_ultra_doppler_list)
    cl_audio = np.append(attack_speech_doppler_list, original_speech_doppler_list)

    
    #shuffle
    length = len(ultra_doppler_list)
    idx = random.sample(range(length), length)
    ultra_doppler_list = ultra_doppler_list[idx]
    speech_doppler_list = speech_doppler_list[idx]
    noisy_speech_doppler_list = noisy_speech_doppler_list[idx]
    #split
    index = int(math.floor(length/hp.batch_size*0.8)*hp.batch_size)
    train_ultra = ultra_doppler_list[:index,]
    test_ultra = ultra_doppler_list[index:,]
    train_noisy = noisy_speech_doppler_list[:index,]
    test_noisy = noisy_speech_doppler_list[index:,]
    train_clean = speech_doppler_list[:index,]
    test_clean = speech_doppler_list[index:,]

    train_denoise_seq = [0] * int(index / hp.batch_size)
    test_denoise_seq = [0] * int((length-index) / hp.batch_size)


    #shuffle
    length = len(cl_label)
    idx = random.sample(range(length), length)
    cl_ultra = cl_ultra[idx]
    cl_audio = cl_audio[idx]
    cl_label = cl_label[idx]
    #split
    index = int(math.floor(length/np.batch_size*0.8)*np.batch_size)
    train_cl_label = cl_label[:index,]
    test_cl_label = cl_label[index:,]
    train_cl_ultra = cl_ultra[:index,]
    test_cl_ultra = cl_ultra[index:,]
    train_cl_audio = cl_audio[:index,]
    test_cl_audio = cl_audio[index:,]
    
    train_cl_seq = [1] * int(index / np.batch_size)
    test_cl_seq = [1] * int((length-index) / np.batch_size)



    #shuffle
    length = len(rg_label)
    idx = random.sample(range(length), length)
    rg_ultra = rg_ultra[idx]
    rg_audio = rg_audio[idx]
    rg_label = rg_label[idx]
    
    #split
    index = int(math.floor(length/hp.batch_size*0.8)*hp.batch_size)
    train_rg_ultra = rg_ultra[:index,]
    test_rg_ultra = rg_ultra[index:,]
    train_rg_audio = rg_audio[:index,]
    test_rg_audio = rg_audio[index:,]
    train_rg_label = rg_label[:index,]
    test_rg_label = rg_label[index:,]

    train_rg_seq = [2] * int(index / np.batch_size)
    test_rg_seq = [2] * int((length-index) / np.batch_size)



    train_seq = train_denoise_seq + train_cl_seq + train_rg_seq
    test_seq = test_denoise_seq + test_cl_seq + test_rg_seq

    train_seq = np.array(train_seq)
    test_seq = np.array(test_seq)

    length = len(train_seq)
    idx = random.sample(range(length), length)
    train_seq = train_seq[idx]
    idx = random.sample(range(length), length)
    train_seq = train_seq[idx]


    save_path1 = "./data/train"
    save_path2 = "./data/test"
    create_file(save_path1)
    create_file(save_path2)

    np.save(save_path1+"/train_ultra.npy", train_ultra)
    np.save(save_path2+"/test_ultra.npy", test_ultra)
    np.save(save_path1+"/train_noisy.npy", train_noisy)
    np.save(save_path2+"/test_noisy.npy", test_noisy)
    np.save(save_path1+"/train_clean.npy", train_clean)
    np.save(save_path2+"/test_clean.npy", test_clean)

    np.save(save_path1+"/train_rg_ultra.npy", train_rg_ultra)
    np.save(save_path2+"/test_rg_ultra.npy", test_rg_ultra)
    np.save(save_path1+"/train_rg_audio.npy", train_rg_audio)
    np.save(save_path2+"/test_rg_audio.npy", test_rg_audio)
    np.save(save_path1+"/train_rg_label.npy", train_rg_label)
    np.save(save_path2+"/test_rg_label.npy", test_rg_label)

    np.save(save_path1+"/train_cl_ultra.npy", train_cl_ultra)
    np.save(save_path2+"/test_cl_ultra.npy", test_cl_ultra)
    np.save(save_path1+"/train_cl_audio.npy", train_cl_audio)
    np.save(save_path2+"/test_cl_audio.npy", test_cl_audio)
    np.save(save_path1+"/train_cl_label.npy", train_cl_label)
    np.save(save_path2+"/test_cl_label.npy", test_cl_label)

    np.save(save_path1+"/train_seq.npy", train_seq)
    np.save(save_path2+"/test_seq.npy", test_seq)
    

    print("train_ultra shape: ", train_ultra.shape)
    print("test_ultra shape: ", test_ultra.shape)
    print("train_rg_ultra shape: ", train_rg_ultra.shape)
    print("test_rg_ultra shape: ", test_rg_ultra.shape)
    print("train_cl_ultra shape: ", train_cl_ultra.shape)
    print("test_cl_ultra shape: ", test_cl_ultra.shape)
    

    return train_ultra, test_ultra, train_noisy, test_noisy, train_clean, test_clean, \
           train_rg_ultra, test_rg_ultra, train_rg_audio, test_rg_audio, train_rg_label, test_rg_label, \
           train_cl_ultra, test_cl_ultra, train_cl_audio, test_cl_audio, train_cl_label, test_cl_label, \
           train_seq, test_seq

def GetDataList(load_exist_list=False):
    save_path1 = "./data/train"
    save_path2 = "./data/test"
    if load_exist_list and os.path.exists(save_path1+"/train_ultra.npy"):
        train_ultra = np.load(save_path1+"/train_ultra.npy")
        test_ultra = np.load(save_path2+"/test_ultra.npy")
        train_noisy =np.load(save_path1+"/train_noisy.npy")
        test_noisy = np.load(save_path2+"/test_noisy.npy")
        train_clean = np.load(save_path1+"/train_clean.npy")
        test_clean = np.load(save_path2+"/test_clean.npy")
        train_rg_ultra = np.load(save_path1+"/train_rg_ultra.npy")
        test_rg_ultra = np.load(save_path2+"/test_rg_ultra.npy", )
        train_rg_audio = np.load(save_path1+"/train_rg_audio.npy", )
        test_rg_audio = np.load(save_path2+"/test_rg_audio.npy", )
        train_rg_label = np.load(save_path1+"/train_rg_label.npy", )
        test_rg_label = np.load(save_path2+"/test_rg_label.npy", )
        train_cl_ultra = np.load(save_path1+"/train_cl_ultra.npy", )
        test_cl_ultra = np.load(save_path2+"/test_cl_ultra.npy", )
        train_cl_audio = np.load(save_path1+"/train_cl_audio.npy", )
        test_cl_audio = np.load(save_path2+"/test_cl_audio.npy", )
        train_cl_label = np.load(save_path1+"/train_cl_label.npy", )
        test_cl_label = np.load(save_path2+"/test_cl_label.npy", )
        train_seq = np.load(save_path1+"/train_seq.npy", )
        test_seq = np.load(save_path2+"/test_seq.npy", )

        return train_ultra, test_ultra, train_noisy, test_noisy, train_clean, test_clean, \
           train_rg_ultra, test_rg_ultra, train_rg_audio, test_rg_audio, train_rg_label, test_rg_label, \
           train_cl_ultra, test_cl_ultra, train_cl_audio, test_cl_audio, train_cl_label, test_cl_label, train_seq, test_seq
    else:
        return GenerateDataList()