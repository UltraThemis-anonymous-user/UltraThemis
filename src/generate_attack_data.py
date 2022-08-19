from ctypes import sizeof
from json import load
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
import config as cfg
from utils.utils import create_file, load_audio
import hyperparameters as hp
from VAD import VAD
import math
import copy


interval = hp.interval
ultra_root = cfg.ultra_path
clean_root = cfg.origin_speech_path
attack_ultra_doppler_path = cfg.attack_ultra_doppler_path
attack_speech_doppler_path = cfg.attack_speech_doppler_path
original_speech_doppler_path = cfg.original_speech_doppler_path
attack_label_path = cfg.attack_label_path

print("Generate replace speech")
ultra_folder_list = os.listdir(ultra_root)
ultra_folder_list.sort()
clean_folder_list = os.listdir(clean_root)
clean_folder_list.sort()

for i in range(len(ultra_folder_list)):

    ultra_folder = ultra_folder_list[i]
    ultra_file_list = os.listdir(ultra_root+ultra_folder+"/"+str(interval))
    ultra_file_list = [file for file in ultra_file_list if os.path.splitext(file)[1] == ".wav"]
    ultra_file_list.sort(key = lambda x: (int(re.split('-|.wav', x)[0]), int(re.split('-|.wav', x)[1])))

    clean_folder = clean_folder_list[i]
    clean_file_list = os.listdir(clean_root+clean_folder+"/"+str(interval))
    clean_file_list = [file for file in clean_file_list if os.path.splitext(file)[1] == ".wav"]
    clean_file_list.sort(key = lambda x: (int(re.split('-|.wav', x)[0]), int(re.split('-|.wav', x)[1])))
    

    path = os.path.join(attack_ultra_doppler_path, ultra_folder, str(interval))
    create_file(path)

    path = os.path.join(attack_speech_doppler_path, ultra_folder, str(interval))
    create_file(path)

    path = os.path.join(original_speech_doppler_path, ultra_folder, str(interval))
    create_file(path)

    path = os.path.join(attack_label_path, ultra_folder, str(interval))
    create_file(path)

    n = 0
    for ultra_file, clean_file in zip(ultra_file_list,clean_file_list):
        
        path = os.path.join(ultra_root, ultra_folder, str(interval), ultra_file)
        ultra_data, ultra_fs = load_audio(path)

        path = os.path.join(clean_root, clean_folder, str(interval), clean_file)
        clean_data, clean_fs = load_audio(path, sr=hp.speech_audio_fs)
        frametime1, voiceseg1, vsl1 = VAD(path)
        if vsl1 == 0:
            continue

        
        
        for repeat in range(10):
            #choose the replace file
            index = random.randint(0, clean_file_list)
            while(clean_file_list[index]==clean_file):
                index = random.randint(0, clean_file_list)
            clean_file2 = clean_file_list[index]
            path = os.path.join(clean_root, clean_folder, str(interval), clean_file2)
            clean_data2, clean_fs = load_audio(path, sr=hp.speech_audio_fs)
            frametime2, voiceseg2, vsl2 = VAD(path)
            if vsl2 == 0:
                continue     
            #choose segment which will be replaced                                                                                                                                                                                                                                                                                       
            index1 = random.randint(0, vsl1-1)

            #choose the replace segment
            index2 = random.randint(0, vsl2-1)

            seg1_start = frametime1[voiceseg1[index1]['start']]
            seg1_end = frametime1[voiceseg1[index1]['end']]
            seg2_start = frametime2[voiceseg2[index2]['start']]
            seg2_end = frametime2[voiceseg2[index2]['end']]

            seg1_start = int(seg1_start*clean_fs)
            seg1_end = int(seg1_end*clean_fs)
            seg2_start = int(seg2_start*clean_fs)
            seg2_end = int(seg2_end*clean_fs)

            if (seg1_end-seg1_start) > (seg2_end-seg2_start):
                dec = int(((seg1_end-seg1_start) - (seg2_end-seg2_start))/2.0)
                inc = int(((seg1_end-seg1_start) - (seg2_end-seg2_start))- dec)
                if seg2_start - dec < 0:
                    inc  += (dec - seg2_start)
                    dec = seg2_start

                if seg2_end + inc > 5*clean_fs:
                    dec += (seg2_end + inc - 5*clean_fs)
                    inc = (5*clean_fs - seg2_end)
                seg2_start -= dec
                seg2_end += inc
            else:
                dec = int(((seg2_end-seg2_start) - (seg1_end-seg1_start))/2.0)
                inc = int(((seg2_end-seg2_start) - (seg1_end-seg1_start))- dec)
                if seg1_start - dec < 0:
                    inc  += (dec - seg1_start)
                    dec = seg1_start

                if seg1_end + inc > 5*clean_fs:
                    dec += (seg1_end + inc - 5*clean_fs)
                    inc = (5*clean_fs - seg1_end)
                seg1_start -= dec
                seg1_end += inc

            
            ultra_final = ultra_data
            clean_final = clean_data
            attack_final = copy.deepcopy(clean_data)
            attack_final[seg1_start:seg1_end] = clean_data2[seg2_start:seg2_end]

            label = [0] * 50
            for i in range(math.floor(seg1_start/clean_fs*10), math.floor((seg1_end-1)/clean_fs*10)+1):
                label[i] = 1

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
            
            path = os.path.join(attack_ultra_doppler_path, ultra_folder, str(interval), ultra_file[:-4]+"-"+str(repeat+1)+".npy")
            np.save(path, feature)

            f, t, zxx = signal.stft(attack_final, fs=clean_fs, nperseg=int(clean_fs*0.032), noverlap=int(clean_fs*0.032 - clean_fs*0.01), nfft=512)
            
            z = abs(zxx[:, ])
            z = (z - z.min()) / (z.max() - z.min())
            feature = np.array(z)
            feature = feature.reshape(1, 1, -1, 501)
            path = os.path.join(attack_speech_doppler_path, ultra_folder, str(interval), ultra_file[:-4]+"-"+str(repeat+1)+".npy")
            np.save(path, feature)
            path = os.path.join(original_speech_doppler_path, ultra_folder, str(interval), ultra_file[:-4]+"-"+str(repeat+1)+".npy")
            np.save(path, feature)
            
            path = os.path.join(attack_label_path, ultra_folder, str(interval), ultra_file[:-4]+"-"+str(repeat+1)+".npy")
            np.save(path, label)

