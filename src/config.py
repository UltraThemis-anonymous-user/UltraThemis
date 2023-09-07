num_threads=2 
wlen = 4096
nfft = 4096
hop = 0.01 #s

############################################################################################################
#######################      parameters used in meta learning       ########################################
# the structure of directory
#   root
#       |____voice(low frequecy part)
#       |       |____person1
#       |               |______1.npy (stft features of a word)
#       |               |______2.npy
#       |       |____person2
#       |       |____.....
#       |____ultra(high frequecy part after demodulation)


normal_data_train = '' # the root path of normal data (with dfs) for training
nodfs_data_train = '' # the root path of normal data (without dfs) for training
normal_data_test = '' # the root path of normal data (with dfs) for test
nodfs_data_test = ''# the root path of normal data (without dfs) for test

############################################################################################################



############################################################################################################
#######################      parameters used in multi-task learning       ##################################
noise_path = " " #path of the noise files
origin_speech_path = " "#path of the ultrasound files
ultra_path = " "#path of the speech files
mixspeech_path = " "#path of the audio with noise
speechAudio_path = " "#path of the audio without noise


ultra_doppler_path = "./data/normal/ultra_doppler/" 
speech_doppler_path = "./data/normal/speech_doppler/"
noisy_speech_doppler_path = "./data/noisy/speech_doppler/"

attack_ultra_doppler_path = "./data/attack/replace/ultra_doppler/"
attack_speech_doppler_path = "./data/attack/replace/speech_doppler/"
original_speech_doppler_path = "./data/attack/replace/original_speech_doppler/"
attack_label_path = "./data/attack/replace/label/"
############################################################################################################


############################################################################################################
#############     parameters used for the model which utilizing both DFS and ANL       #####################

# the structure of directory
#   root
#       |____attack type(e.g L1H2)
#               |____voice(low frequecy part)
#               |       |____person1
#               |               |______1.npy (stft features of a word)
#               |               |______2.npy
#               |       |____person2
#               |       |____.....
#               |____ultra(high frequecy part after demodulation)

root_path_train = ""# the root path of training data
root_path_test = ""

# the type that will be treated as a positive sample，
# normal: untampered data ;  distance: data record in different distance
# noise: data with varying degrees of background noise 
# phone: data recorded by different phones
# hold: different holding styles
# direction: data recorded in different directions
# move: different movement states of the handholder
positive_data_type = ["normal", "distance", "noise", "phone", "hold", "direction", "move"]

###########################################################################################################

