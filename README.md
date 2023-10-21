# UltraThemis
UltraThemis is a novel trustworthy speech acquisition with tamper-proof detection. We are the first to probe the acoustic nonlinearity effect that reversely converts audible signals into ultrasonic bands, which has not been explored in the literature, and exploit multiple acoustic effect for ultrasonic sensing. UltraThemis enables a proactive and scalable detection against various tampering methods.

<strong>Doppler Frequency Shift (DFS)</strong> and <strong>Time-Of-Flight (TOF)</strong> are often used to describe the dynamic and static characteristics of vocal tract in ultrasound sensing techniques, which bring the correlation between speech signal and ultrasound. Additionally, we discover a new acoustic nonlinear phenomenon
where audible signals can be modulated onto the ultrasonic spectrum due to microphone nonlinearity. <strong>The acoustic nonlinear (ANL)</strong> characteristic further strengthens the correlation between speech signal and ultrasound. Based on these three ultrasound effects, we propose UltraThemis, a tamper-proof speech acquisition system, which demands neither peripheral nor modification on smartphone-like mobile devices. Ultrasound is transmitted by the build-in speaker and received by the build-in microphone.  In particular, we design a meta learning based method to improve the performance of UltraThemis upon new users without the requirement of registration. This repository includes the code which realizes audio tamper detection based on only DFS or both ANL and DFS, and the result of the system on different mobile phones. We will release our datasets and complete code to facilitate the ultrasonic sensing research after necessary data desensitization according to the requirements of IBR and relevant regulations.  

# Repository structure
### src  
+ This folder contains the source code 
+ MTL: source code which realizes audio tamper detection based on only DFS in case that some smartphones presents no obvious nonlinearity.    
+ two-stream: source code which realizes audio tamper detection based on both DFS and ANL. The basic model used in content-consistency network, homology network and meta-learning is ResNet18.  


### The performance on diverse smartphones.md  
+ This file records the result of our system on different mobile phones, including accuracy, precision and recall.  

# The architecture of ResNet18
The ResNet18 is consist of 17 convolutional layers and 1 fully connected layer. The concrete parameters is shown below. For each row, (N×N, C) means the kernel size is N and the number of output channel is C.

![resnet18](./resnet18.png "The architecture of ResNet18")

# Installation
Ensure you have:  
+ Python >= 3.7
+ Pytorch 1 with CUDA
+ librosa  

# How to use
Here's my recommendation on what order to run things:

<strong>MTL</strong>(realizes audio tamper detection based on only DFS):

1 - prepare your dataset and alter the parameter in src/config.py

2 - generate noisy audios

> python src/MTL/add_noise.py

3 - generate tampered audios:

> python src/MTL/generate_attack_data.py

4 - train your own model:

> python src/MTL/train_MTL.py

<strong>two-stream</strong>(realizes audio tamper detection based on both DFS and ANL):

1 - prepare your dataset and alter the parameter in src/config.py

2 - preprocess data for meta-learning training

> python src/two-stream/meta-learning/meta_data_preprocess.py

3 - train meta-learning model:

> python src/two-stream/meta-learning/train_meta.py

Here are the hyper-parameters that you can setup
+ model_save_path: the path where the model will be saved
+ epoch: the number of epoches the model will be trained
+ n_way: in every task, the samples will consist of data from n_way individuals
+ k_spt: k shot for support set
+ k_qry: k shot for query set
+ batch_size: batch size
+ task_num: meta batch size
+ meta_lr: meta-level outer learning rate
+ update_lr: task-level inner update learning rate
+ update_step: task-level inner update steps
+ update_step_test: update steps for finetunning

4 - preprocess data for content-consistency network and homology network training

> python src/two-stream/data_preprocess.py

5 - train content-consistency network and homology network:

> python src/train.py

Here are the hyper-parameters that you can setup
+ model_save_path: the path where the model will be saved
+ anl_or_dfs: train content-consistency network or homology network, 0 for the former and 1 for the latter
+ epoch: the number of epoches the model will be trained
+ batch_size: batch size
+ learning_rate: learning rate
+ is_reload_model: load pre-trained model or not

6 - test overall performance
> python src/test.py

Here are the hyper-parameters that you can setup
+ batch_size: batch size
+ model_save_path1: the path where the content-consistency network saved
+ model_save_path2: the path where the homology network saved
+ model_save_path3: the path where the meta-learning network saved
+ threshold1: the threshold for content-consistency network
+ threshold2: the threshold for homology network
+ threshold3: the threshold for meta-learning network

# Parameter Settings

Parameters of STFT
+ window size: 4096
+ nfft : 4096
+ hop length: 10ms
+ the length of every audio segment: 0.5s

Parameters of meta-learning
+ epoch: 60000
+ n_way: 5
+ k_spt: 4
+ k_qry: 10
+ batch_size: 8
+ task_num: 4
+ meta_lr: 1e-3
+ update_lr: 0.01
+ update_step: 5
+ update_step_test: 10
+ optimizer: Adam

Parameters of two-stream
+ epoch: 50
+ batch_size: 64
+ learning_rate: 1e-4
+ optimizer: Adam
+ scheduler: StepLR(step_size=5, gamma=0.25)
+ threshold1: 0.8
+ threshold2: 0.4
+ threshold3: 0.4


