# UltraThemis
UltraThemis is a novel trustworthy speech acquisition with tamper-proof detection. We are the first to probe the acoustic nonlinearity effect that reversely converts audible signals into ultrasonic bands, which has not been explored in the literature, and exploit multiple acoustic effect for ultrasonic sensing. UltraThemis enables a proactive and scalable detection against various tampering methods.

<strong>Doppler Frequency Shift (DFS)</strong> and <strong>Time-Of-Flight (TOF)</strong> are often used to describe the dynamic and static characteristics of vocal tract in ultrasound sensing techniques, which bring the correlation between speech signal and ultrasound. Additionally, we discover a new acoustic nonlinear phenomenon
where audible signals can be modulated onto the ultrasonic spectrum due to microphone nonlinearity. <strong>The acoustic nonlinear (ANL)</strong> characteristic further strengthens the correlation between speech signal and ultrasound. Based on these three ultrasound effects, we propose UltraThemis, a tamper-proof speech acquisition system, which demands neither peripheral nor modification on smartphone-like mobile devices. Ultrasound is transmitted by the build-in speaker and received by the build-in microphone.  In particular, we design a meta learning based method to improve the performance of UltraThemis upon new users without the requirement of registration. This repository includes the code which realizes audio tamper detection based on only DFS and the result of the system on different mobile phones. We will release our datasets and complete code to facilitate the ultrasonic sensing research after necessary data desensitization according to the requirements of IBR and relevant regulations.  

# Repository structure
### src  
+ This folder contains the source code which realizes audio tamper detection based on only DFS in case that some smartphones presents no obvious nonlinearity.    
### The performance on diverse smartphones.md  
+ This file records the result of our system on different mobile phones, including accuracy, precision and recall.  

# Installation
Ensure you have:  
+ Python >= 3.7
+ Pytorch 1 with CUDA
+ librosa  

# How to use
Here's my recommendation on what order to run things:

1 - prepare your dataset and alter the parameter in config.py

2 - generate noisy audios

> python add_noise.py

3 - generate tampered audios:

> python generate_attack_data.py

4 - train your own model:

> python train.py
