import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal as signal
from VAD2 import VAD
import os
import config as cfg
import random
import librosa
import matplotlib
import sys
import shutil
np.seterr(divide='ignore', invalid='ignore')

def create_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("remove "+path)

    os.makedirs(path)
    print("create "+path)



mode = 5#0-clean; 1-replace; 2-synthesize; 3-immate; 4-nodoppler;5-others
isCreateFilePath=1
pickNameList = []#'cyk', 'fxy',  "ljw", , , "zx""gwt", "slm", "swf", "xr", "xwy", "zlf", "cyk"
banNameList = ['hyh']

if mode==0:
    output_voice_root = cfg.segment_clean_doppler
    output_ultra_root = cfg.segment_clean_ultra_doppler
    root_high_path = os.path.join(cfg.origin_ultra_path, "clean")
    root_low_path = os.path.join(cfg.origin_speech_path, "clean")
elif mode==1:
    output_voice_root = cfg.segment_replace_doppler
    output_ultra_root = cfg.segment_replace_ultra_doppler
    root_high_path = os.path.join(cfg.origin_ultra_path, "clean")
    root_low_path = os.path.join(cfg.origin_speech_path, "clean")
elif mode==2:
    output_voice_root = cfg.segment_synthesize_doppler
    output_ultra_root = cfg.segment_synthesize_ultra_doppler
    root_high_path = os.path.join(cfg.origin_ultra_path, "synthesize")
    root_low_path = os.path.join(cfg.origin_speech_path, "synthesize")
elif mode==3:
    output_voice_root = cfg.segment_immate_doppler
    output_ultra_root = cfg.segment_immate_ultra_doppler
    root_high_path = os.path.join(cfg.origin_ultra_path, "immate")
    root_low_path = os.path.join(cfg.origin_speech_path, "immate")
elif mode==4:
    output_voice_root = cfg.segment_wo_doppler
    output_ultra_root = cfg.segment_wo_ultra_doppler
    root_high_path = os.path.join(cfg.origin_ultra_path, "nodoppler")
    root_low_path = os.path.join(cfg.origin_speech_path, "nodoppler")
elif mode==5:
    output_voice_root = cfg.root_path+cfg.phone_type+cfg.dataset+"/direction/doppler/"
    output_ultra_root = cfg.root_path+cfg.phone_type+cfg.dataset+"/direction/ultra_doppler/"
    root_high_path = os.path.join(cfg.origin_ultra_path, "direction")
    root_low_path = os.path.join(cfg.origin_speech_path, "direction")
else:
    sys.exit(0)


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
        if (mode==0 or mode==1 or mode==3 or mode==4):
            if not (int(file[:-4])%20>=1 and int(file[:-4])%20<=16):
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
        
        # plt.figure(figsize=(50,20))
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
        # plt.subplot(2, 1, 1)
        # plt.pcolormesh((abs(z_high[2:int(2000/resolution), :-1])))
        # plt.subplot(2, 1, 2)
        # plt.pcolormesh((abs(z_low[2:int(2000/resolution), :-1])))

        # plt.show()
        
        
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
            
            # plt.figure(figsize=(5,20))
            # norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
            # plt.subplot(2, 1, 1)
            # plt.pcolormesh((ultra_doppler[0]))
            # plt.subplot(2, 1, 2)
            # plt.pcolormesh((voice_doppler[0]))

            # plt.show()
            # c = input()
            # plt.close()
            # if c=='d':
            #     continue

            np.save(save_path1, voice_doppler)
            np.save(save_path2, ultra_doppler)

        

    print()




            



    