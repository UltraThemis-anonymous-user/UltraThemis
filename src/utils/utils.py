import os
import shutil
import librosa


def create_file(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("remove "+path)

    os.makedirs(path)
    print("create "+path)

def load_audio(path, target_fs=None):
    data, fs = librosa.load(path, sr=None)
    print(path, end='\r')
    if target_fs!=None and fs != target_fs:
        data = librosa.resample(data,fs,target_fs)
        fs = target_fs
    return data, fs