import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal as signal
import sys
sys.path.append("../")
from VAD2 import VAD
import os
import config as cfg
import random
import librosa
import matplotlib

import shutil
np.seterr(divide='ignore', invalid='ignore')

def create_file(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("create "+path)
    else:
        shutil.rmtree(path)
        print("remove "+path)



mode = 0#0-clean; 1-replace; 2-synthesize; 3-immate; 4-nodoppler; 5-different people
isCreateFilePath=1
pickNameList = []#'cyk', 'fxy',  "ljw", , , "zx""gwt", "slm", "swf", "xr", "xwy", "zlf", "cyk"
banNameList = ['hyh', 'zlf_slm']

output_voice_root = os.path.join(cfg.root_path, cfg.phone_type, "test_meta","nodoppler" ,"voice")
output_ultra_root = os.path.join(cfg.root_path, cfg.phone_type, "test_meta","nodoppler" ,"ultra")
root_high_path = os.path.join(cfg.origin_ultra_path, "clean")
root_low_path = os.path.join(cfg.origin_speech_path, "clean")



folder_list = os.listdir(root_high_path)
folder_list.sort()
create_file(output_voice_root)
create_file(output_ultra_root)

for folder in folder_list:
    if len(pickNameList)!=0 and folder not in pickNameList:
        continue
    if folder in banNameList:
        continue

    if isCreateFilePath:
        path = os.path.join(output_voice_root, folder)
        create_file(path)
        path = os.path.join(output_ultra_root, folder)
        create_file(path)
        
    file_list = os.listdir(os.path.join(root_high_path, folder))
    file_list.sort(key = lambda x: (int(x[:-4])))
    for file in file_list:
        if int(file[:-4])<=40:
            continue
        if (int(file[:-4])%20>=1 and int(file[:-4])%20<=16):
            continue
 
        path1 = os.path.join(root_high_path, folder, file)
        path2 = os.path.join(root_low_path, folder, file)
        
        high, fs=librosa.load(path1, sr=48000)
        low, fs=librosa.load(path2, sr=48000)
        
        wlen=4096
        nfft=wlen
        hop=int(fs*0.01)
        resolution=fs/nfft
        f, t, z_high = signal.stft(high, fs=fs, nperseg=wlen, noverlap=(wlen-hop), nfft=nfft)
        f, t, z_low = signal.stft(low, fs=fs, nperseg=wlen, noverlap=(wlen-hop), nfft=nfft)

        frametime, voiceseg, vsl = VAD(path2)
        
        begin_points = []
        end_points = []
        max_length=int(0.5*fs)
        for i in range(vsl):            
            if frametime[voiceseg[i]['end']] - frametime[voiceseg[i]['start']]<0.1:
                continue
            mid = int((int(frametime[voiceseg[i]['end']]*fs)+int(frametime[voiceseg[i]['start']]*fs))/2)
            begin = mid-int(0.25*fs)
            end = mid+int(0.25*fs)
            if end>int(len(low)) or begin<0:
                continue
            begin_points += [begin]
            end_points += [end]

        vsl = len(begin_points)
        print(path1, vsl, end='\r')

        for i in range(vsl):
            j=random.randint(0, vsl-1)
            while(i==j):
                j=random.randint(0, vsl-1)
            segment = high[begin_points[i]:end_points[i]]
            segment = segment/np.max(np.abs(segment))
            f, t, z_high = signal.stft(segment, fs=fs, nperseg=wlen, noverlap=(wlen-hop), nfft=nfft)

            if mode==1:
                segment = low[begin_points[j]:end_points[j]]
                segment = segment/np.max(np.abs(segment))
                f, t, z_low = signal.stft(segment, fs=fs, nperseg=wlen, noverlap=(wlen-hop), nfft=nfft)
            else:
                segment = low[begin_points[i]:end_points[i]]
                segment = segment/np.max(np.abs(segment))
                f, t, z_low = signal.stft(segment, fs=fs, nperseg=wlen, noverlap=(wlen-hop), nfft=nfft)

            

            voice_doppler = abs(z_low[2:int(2000/resolution), :-1])
            ultra_doppler = abs(z_high[2:int(2000/resolution), :-1])

            voice_doppler = voice_doppler/np.max(voice_doppler)
            ultra_doppler = ultra_doppler/np.max(ultra_doppler)


            save_path1 = os.path.join(output_voice_root, folder, os.path.splitext(file)[0]+"-"+str(i+1)+"-"+str(mode)+".npy")
            voice_doppler = np.reshape(voice_doppler, (1, voice_doppler.shape[0], voice_doppler.shape[1]))
            
            save_path2 = os.path.join(output_ultra_root, folder, os.path.splitext(file)[0]+"-"+str(i+1)+"-"+str(mode)+".npy")
            ultra_doppler = np.reshape(ultra_doppler, (1, ultra_doppler.shape[0], ultra_doppler.shape[1]))
            assert voice_doppler.shape == (1, 168, 50)
            assert ultra_doppler.shape == (1, 168, 50)
            

            np.save(save_path1, voice_doppler)
            np.save(save_path2, ultra_doppler)

        

    print()




            



    