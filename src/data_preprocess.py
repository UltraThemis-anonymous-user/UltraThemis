import numpy as np
import os
import config as cfg
import re
import soundfile as sf
import scipy.signal as signal
from scipy.io import wavfile
import librosa
import random
import matplotlib.pyplot as plt
import math
from utils.utils import create_file, load_audio
import hyperparameters as hp
import glob

def compute_ultra_doppler(ultra_data, ultra_fs):
    f, t, zxx = signal.stft(ultra_data, fs=ultra_fs, nperseg=int(ultra_fs*0.085), noverlap=int(ultra_fs*0.085 - ultra_fs*0.01), nfft=hp.nfft)
    feature = []
    resolution = ultra_fs/hp.nfft
    for i in range(0, 8):
        fre = 16000 + 1000*i
        
        index = fre // resolution
        f_t = f[index]
        
        f_t = f[index-int(400/resolution):index+int(400/resolution)]
        z = zxx[index-int(400/resolution):index+int(400/resolution)]
        z = abs(z)
        z = (z - z.min()) / (z.max() - z.min())

        z = np.array(z)
        z = z.reshape(1, -1, 501)
        
        
        if i==0:
            feature = z
        else:
            feature = np.concatenate((feature, z), 0)
    feature = feature.reshape(1, 8, -1, 501)
    return feature


ultra_file = cfg.ultra_path
clean_file = cfg.speechAudio_path

ultra_doppler_path = "./data/normal/ultra_doppler/" 
speech_doppler_path = "./data/normal/speech_doppler/"
interval = hp.interval

print("preprocess ultrasound")

folder_list = os.listdir(ultra_file)
folder_list.sort()
for folder in folder_list:
    T_F_features = []
   
    file_list = glob.glob(os.path.join(ultra_file, folder, str(interval)), "*.wav")
    file_list.sort(key = lambda x: (int(re.split('-|.wav', x)[0]), int(re.split('-|.wav', x)[1])))

    create_file(os.path.join(ultra_doppler_path, folder, str(interval)))
    num = 1
    is_begin = 1
    for file in file_list:
        if num == 4:
            num = 1
  
        path = os.path.join(ultra_file, folder, str(interval), file) 
        ultra_data, ultra_fs = load_audio(path)
        feature = compute_ultra_doppler(ultra_data, ultra_fs)
        np.save(cfg.ultra_doppler_path+folder+"/"+str(interval)+"/"+ file[:-4]+".npy", feature)

print("preprocess clean speech")
num = 0
index = 0
is_begin = 1
folder_list = os.listdir(clean_file)
folder_list.sort()
for folder in folder_list:
    file_list = os.listdir(clean_file+folder+"/"+str(interval))
    file_list = [file for file in file_list if os.path.splitext(file)[1] == ".wav"]
    file_list.sort(key = lambda x: (int(re.split('-|.wav', x)[0]), int(re.split('-|.wav', x)[1]), int(re.split('-|.wav', x)[2])))

    create_file(cfg.speech_doppler_path+folder+"/"+str(interval))

    for file in file_list:
        path = clean_file+folder+"/"+str(interval)+"//"+file
        num += 1
  
        data, fs = load_audio(path, hp.speech_audio_fs)

        f, t, zxx = signal.stft(data, fs=fs, nperseg=int(fs*0.032), noverlap=int(fs*0.032 - fs*0.01), nfft=512)
        
        
        z = abs(zxx[:, ])
        z = (z - z.min()) / (z.max() - z.min())
        feature = np.array(z)
        feature = feature.reshape(1, 1, -1, 501)

        
        np.save(cfg.speech_doppler_path+folder+"/"+str(interval)+"//"+file[:-4]+".npy", feature)
  
      

print("preprocess noisy speech")

noisy_speech_doppler_path = cfg.noisy_speech_doppler_path

noisy_file = cfg.mixspeech_path
folder_list = os.listdir(noisy_file)
folder_list.sort()
is_begin = 1
count = 0
index = 0
for folder in folder_list:
    file_list = glob.glob(os.path.join(noisy_file, folder, str(interval)), "*.wav")
    file_list.sort(key = lambda x: (int(re.split('-|-|.wav', x)[0]), int(re.split('-|-|.wav', x)[1]), int(re.split('-|-|.wav', x)[2])))

    create_file(cfg.noisy_speech_doppler_path+folder+"/"+str(interval))


    for file in file_list:

        path = noisy_file+folder+"/"+str(interval)+"/"+file
        count += 1
        data, fs = sf.read(path)
  
        print(path, end='\r')


        data, fs = load_audio(path, hp.speech_audio_fs)

        f, t, zxx = signal.stft(data, fs=fs, nperseg=int(fs*0.032), noverlap=int(fs*0.032 - fs*0.01), nfft=512)
        
        
        z = abs(zxx[:, ])
        z = (z - z.min()) / (z.max() - z.min())
        feature = np.array(z)
        feature = feature.reshape(1, 1, 257, -1)
       
        np.save(cfg.noisy_speech_doppler_path+folder+"/"+str(interval)+"/"+file[:-4]+".npy", feature)




  

        
        

