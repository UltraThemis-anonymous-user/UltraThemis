from inspect import FrameInfo
import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.ma.core import concatenate
import scipy.signal as signal
import os
import scipy.io as scio
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import random
import soundfile as sf
import time
import math
import scipy.io
import config as cfg
import re
from utils.utils import create_file, load_audio
import hyperparameters as hp
import glob


def rootMeanSquare(y):

    return np.sqrt(np.mean(np.abs(y) ** 2 ,axis = 0 ,keepdims = False))

def getAmplitudeScalingFactor(s,n,snr,method="rms"):

    originalRmsRatio = rootMeanSquare(s) / rootMeanSquare(n)
    targetRmsRatoio = 10.** (float(snr)/20.)
    signalScalingFactor = targetRmsRatoio/originalRmsRatio
    return signalScalingFactor

def additiveMixing(s,n):


    mixedAudio = s + n
    alpha = 1. / np.max(np.abs(mixedAudio))
    # mixedAudio *= alpha
    # s *= alpha
    # n *= alphapytho
    return mixedAudio,s,n,alpha



def computeSNR(s,n):
    # n = n / np.max(np.abs(n))
    ratio = rootMeanSquare(s) / rootMeanSquare(n)
    snr = 20. * math.log10(ratio)
    return snr


def add_noise(speech_data, snr=10):

    #get the list of noise files
    noise_list = glob.glob(noise_path, "*.wav")
    noise_list.sort(key = lambda x: int(x[1:-4]))

    noise_index = random.randint(0, len(noise_list)-1)

    
    noise_data, fs = load_audio(noise_path+"//"+noise_list[noise_index], fs=16000)
    noise_data = noise_data + 0.
  
    if len(noise_data) < len(speech_data):
        times = int(np.ceil(len(speech_data)* 1. / len(noise_data)))
        noise_data = np.tile(noise_data, times)
    noise_data = noise_data[0:len(speech_data)]
    noise_data = np.frombuffer(noise_data, dtype=noise_data.dtype)
    noise_data = noise_data.copy()

    scaler = getAmplitudeScalingFactor(speech_data, noise_data, snr)
    speech_data *= scaler

    (mixedAudio,speechAudio,noiseAudio,alpha) = additiveMixing(speech_data,noise_data)

    return mixedAudio, speechAudio


if __name__ =="__main__":
    print("Generating noisy speech")

    noise_path = cfg.noise_path
    speech_path = cfg.origin_speech_path
    interval = hp.interval

    

    #get the list of the clean audio files
    speech_folder_list = glob.glob(speech_path, "*.wav")

    noise_pseech = []
    snrs = []
    for speech_folder in speech_folder_list:

        speech_folder_path = os.path.join(speech_path, speech_folder, str(interval))
        speech_list = os.listdir(speech_folder_path)
        speech_list.sort(key = lambda x: (int(re.split('-|.wav', x)[0]), int(re.split('-|.wav', x)[1])))

        mixed_save_path = os.path.join(cfg.mixspeech_path, speech_folder, str(interval) )
        speech_save_path = os.path.join(cfg.speechAudio_path, speech_folder, str(interval))
        
        create_file(mixed_save_path)
        create_file(speech_save_path)

        for speech in speech_list:
    
            speech_data, fs = load_audio(speech_folder_path+"//"+speech, hp.speech_audio_fs) 

            speech_data = speech_data + 0.


            if len(speech_data) > fs*5:
                speech_data = speech_data[len(speech_data)-fs*5:]

            for count in range(hp.noisy_audio_num):
                snr = random.uniform(-9, 6)
                mixedAudio,speechAudio = add_noise(speech_data, snr)

                filename = os.path.join("%s-%d.wav" %(os.path.splitext(speech)[0], count))
                snrs = snrs + [snr]
                sf.write(cfg.mixspeech_path+speech_folder+"/"+str(interval)+"/"+filename, mixedAudio, fs)
                sf.write(cfg.speechAudio_path+speech_folder+"/"+str(interval)+"/"+filename, speechAudio, fs)
         
       
    scipy.io.savemat("./snrs.mat", {'snr': snrs})
    print("over!")


