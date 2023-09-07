from json import load
import numpy as np
import torch
from model import Encoder_Block, Residual_Block, Decoder_Block, Dricriminator1, Dricriminator2
import config as cfg
import os 
import re
import random
import math
import torch.nn as nn
from torch.autograd.variable import Variable
from numpy.core.fromnumeric import argmax
import matplotlib.pyplot as plt
import hyperparameters as hp
from GenerateDataList import GetDataList

interval=cfg.interval
print(interval)
model_save_path = "./model"+str(interval)+"/model.pkl"
batch_size = hp.batch_size
total_epoch = hp.total_epoch
learning_rate = hp.learning_rate
load_model = False

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def ReadData(list):
    data = []
    is_begin = 1
    for file in list:
        temp = np.load(file)
        if is_begin:
            data = temp
            is_begin = 0
        else:
            data = np.concatenate((data, temp), 0)
    return data
def ReadLabel(list):
    data = []
    is_begin = 1
    for file in list:
        temp = np.load(file)
        temp = np.reshape(temp, [1, -1])
        if is_begin:
            data = temp
            is_begin = 0
        else:
            data = np.concatenate((data, temp), 0)
    return data

if __name__ == "__main__":

    train_ultra, test_ultra, train_noisy, test_noisy, train_clean, test_clean, \
    train_rg_ultra, test_rg_ultra, train_rg_audio, test_rg_audio, train_rg_label, test_rg_label, \
    train_cl_ultra, test_cl_ultra, train_cl_audio, test_cl_audio, train_cl_label, test_cl_label, train_seq, test_seq = GetDataList()

    device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
    print(device)
    Encoder = Encoder_Block().to(device)
    ResNet = Residual_Block().to(device)
    Denoise = Decoder_Block().to(device)
    Discriminator = Dricriminator1().to(device)
    Regression = Dricriminator2().to(device)

    opt1 = torch.optim.Adam([
            {'params': Encoder.parameters(), 'lr': learning_rate,}, 
            {'params': ResNet.parameters()},
            {'params': Denoise.parameters()},
            ])
    scheduler1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=5, gamma=0.25)


    opt2 = torch.optim.Adam([
            {'params': Encoder.parameters(), 'lr': learning_rate,}, 
            {'params': ResNet.parameters()},
            {'params': Discriminator.parameters()},
            ])
    scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=5, gamma=0.25)

    opt3 = torch.optim.Adam([
            {'params': Encoder.parameters(), 'lr': learning_rate,}, 
            {'params': ResNet.parameters()},
            {'params': Regression.parameters()},
            ])
    scheduler3 = torch.optim.lr_scheduler.StepLR(opt3, step_size=5, gamma=0.25)

    epoch_start = 0
    if load_model:
        ckpt = torch.load(model_save_path)
        Encoder.load_state_dict(ckpt['model_Enc_state_dict'])
        ResNet.load_state_dict(ckpt['model_Res_state_dict'])
        Denoise.load_state_dict(ckpt['model_Den_state_dict'])
       
        Discriminator.load_state_dict(ckpt['model_Dis_state_dict'])
        Regression.load_state_dict(ckpt['model_Reg_state_dict'])

        opt1.load_state_dict(ckpt['optimizer_1_state_dict'])
        opt2.load_state_dict(ckpt['optimizer_2_state_dict'])
        opt3.load_state_dict(ckpt['optimizer_3_state_dict'])

        scheduler1.load_state_dict(ckpt['scheduler_1_state_dict'])
        scheduler2.load_state_dict(ckpt['scheduler_2_state_dict'])
        scheduler3.load_state_dict(ckpt['scheduler_3_state_dict'])

        epoch_start = ckpt['epoch']
        print(epoch_start)


    loss1 = nn.MSELoss(reduction='mean')
    loss2 = nn.CrossEntropyLoss()

    

    task_name = ["denoise", "cl", "rg"]
    for epoch in range(epoch_start+1, total_epoch+1):
        
        #train
        loss_Denoise_total = 0
        loss_Recover_total = 0
        loss_Discriminator_total = 0
        loss_Regression_total = 0
        right_num = 0
        num = 0
        task1_index = 0 #denoise
        task2_index = 0 #cl
        task3_index = 0 #rg
        for seq in train_seq:   
            num += 1
            # seq = 0
            if seq!=0 and seq!=2:
               continue
            if seq==0:
                ultrasound = torch.from_numpy(ReadData(train_ultra[task1_index:task1_index+batch_size,]))
                clean = torch.from_numpy(ReadData(train_clean[task1_index:task1_index+batch_size,]))
                audio = torch.from_numpy(ReadData(train_noisy[task1_index:task1_index+batch_size,]))
                
                ultrasound = Variable(ultrasound.to(torch.float32), requires_grad=True).to(device)
                audio = Variable(audio.to(torch.float32), requires_grad=True).to(device)
                clean = Variable(clean.to(torch.float32), requires_grad=True).to(device)

                task1_index += batch_size
            elif seq==1:
                ultrasound = torch.from_numpy(ReadData(train_cl_ultra[task2_index:task2_index+batch_size,]))
                audio = torch.from_numpy(ReadData(train_cl_audio[task2_index:task2_index+batch_size,]))
                label = torch.from_numpy(train_cl_label[task2_index:task2_index+batch_size])

                ultrasound = Variable(ultrasound.to(torch.float32), requires_grad=True).to(device)
                audio = Variable(audio.to(torch.float32), requires_grad=True).to(device)
                label = Variable(label.to(torch.float32), requires_grad=True).to(device)
                task2_index += batch_size
                pass
            elif seq==2:
                ultrasound = torch.from_numpy(ReadData(train_rg_ultra[task3_index:task3_index+batch_size,]))
                audio = torch.from_numpy(ReadData(train_rg_audio[task3_index:task3_index+batch_size,]))
                label = torch.from_numpy(ReadLabel(train_rg_label[task3_index:task3_index+batch_size,]))

                ultrasound = Variable(ultrasound.to(torch.float32), requires_grad=True).to(device)
                audio = Variable(audio.to(torch.float32), requires_grad=True).to(device)
                label = Variable(label.to(torch.float32), requires_grad=True).to(device)
                task3_index += batch_size
                
                pass

            
            out = Encoder(ultrasound, audio)
            out = ResNet(out)

            if seq==0:
                out = Denoise(out)*audio
                loss = loss1(out, clean)*10
                loss_Denoise_total += loss.item()
                opt = opt1
                pass
            elif seq==1:
                out = Discriminator(out)
                loss = loss2(out, label.long())
                loss_Discriminator_total += loss.item()
                pred = torch.argmax(out, dim=1)
                right_num += sum(pred==label).item()
                opt = opt2
                pass
            elif seq==2:
                out = Regression(out)
                loss = loss1(out, label)*10
                loss_Regression_total += loss.item()
                opt = opt3
                pass
  
            opt.zero_grad()
            loss.backward()
            opt.step()
    
   
            print("train: %d/%d, task: %s, D loss: %f"% (num, len(train_seq), task_name[seq], loss.item()), end='\r')
          
        print("train: %d/%d, task: %s, loss: %f, task: %s, loss: %f, acc: %f, task: %s, loss: %f"% (epoch, total_epoch, \
                                                    task_name[0], loss_Denoise_total/(len(train_ultra)/batch_size), \
                                                    task_name[2], loss_Discriminator_total/(len(train_cl_ultra)/batch_size), right_num/len(train_cl_ultra),\
                                                    task_name[3], loss_Regression_total/(len(train_rg_ultra)/batch_size),))
       
        #test
        loss_Denoise_total = 0
        loss_Discriminator_total = 0
        loss_Regression_total = 0
        right_num = 0
        num = 0
        task1_index = 0 #denoise
        task2_index = 0 #cl
        task3_index = 0 #rg

        FP_num = 0
        N_num = 0
        area = 0
        no_overlap_num = 0
        overlap_num = 0
        part_overlap_num = 0
        in_overlap_num = 0
        out_overlap_num = 0
        
        count = 0
        for seq in test_seq:   
            num += 1
         
            if seq==0:
                ultrasound = torch.from_numpy(ReadData(test_ultra[task1_index:task1_index+batch_size,]))
                clean = torch.from_numpy(ReadData(test_clean[task1_index:task1_index+batch_size,]))
                audio = torch.from_numpy(ReadData(test_noisy[task1_index:task1_index+batch_size,]))
                
                ultrasound = Variable(ultrasound.to(torch.float32), requires_grad=False).to(device)
                audio = Variable(audio.to(torch.float32), requires_grad=False).to(device)
                clean = Variable(clean.to(torch.float32), requires_grad=False).to(device)

                task1_index += batch_size
                pass
            elif seq==2:
                ultrasound = torch.from_numpy(ReadData(test_cl_ultra[task2_index:task2_index+batch_size,]))
                audio = torch.from_numpy(ReadData(test_cl_audio[task2_index:task2_index+batch_size,]))
                label = torch.from_numpy(test_cl_label[task2_index:task2_index+batch_size,])

                ultrasound = Variable(ultrasound.to(torch.float32), requires_grad=False).to(device)
                audio = Variable(audio.to(torch.float32), requires_grad=False).to(device)
                label = Variable(label.to(torch.float32), requires_grad=False).to(device)
                task2_index += batch_size
                pass
            elif seq==3:
                ultrasound = torch.from_numpy(ReadData(test_rg_ultra[task3_index:task3_index+batch_size,]))
                audio = torch.from_numpy(ReadData(test_rg_audio[task3_index:task3_index+batch_size,]))
                label = torch.from_numpy(ReadLabel(test_rg_label[task3_index:task3_index+batch_size,]))

                ultrasound = Variable(ultrasound.to(torch.float32), requires_grad=False).to(device)
                audio = Variable(audio.to(torch.float32), requires_grad=False).to(device)
                label = Variable(label.to(torch.float32), requires_grad=False).to(device)
                task3_index += batch_size
                pass

            
            out = Encoder(ultrasound, audio)
            out = ResNet(out)

            if seq==0:
                out = Denoise(out)*audio
                loss = loss1(out, clean)
                loss_Denoise_total += loss.item()
                out = out.cpu().detach().numpy()
            elif seq==2:
                out = Discriminator(out)
                loss = loss2(out, label.long())
                loss_Discriminator_total += loss.item()
                pred = torch.argmax(out, dim=1)
                right_num += sum(pred==label).item()
                for i in range(len(pred)):
                    if label[i]==1:
                        N_num += 1
                    if pred[i]==0 and label[i]==1:
                        FP_num += 1
            
            elif seq==3:
                out = Regression(out)
                loss = loss1(out, label)
                loss_Regression_total += loss.item()


                out = out.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                begin_real = argmax(label, 1)
                end_real = []
                for i in range(batch_size):
                    for j in range(len(label[i])-1, -1, -1):
                        if label[i][j] == 1:
                            end_real += [j]
                            break
                
                out[out>0.8] = 1
                out[out<=0.8] = 0
                begin_out = argmax(out, 1)
                end_out = []
                
                for i in range(batch_size):
                    for j in range(len(out[i])-1, -1, -1):
                        if out[i][j] == 1:
                            end_out += [j]
                            break
                        if j==0:
                            end_out += [begin_out[i]]
                
                for i in range(batch_size):
                    if begin_out[i]>end_out[i]:
                        temp = begin_out[i]
                        begin_out[i] = end_out[i]
                        end_out[i] = begin_out[i]
                    inner_area = 0
                    begin = max([begin_out[i], begin_real[i]])
                    end = min([end_out[i], end_real[i]])
                    if end_real[i]!=begin_real[i]:
                        inner_area = (end-begin)/(end_real[i]-begin_real[i])
                    if inner_area>0:
                        area += inner_area
                    
                    if end_out[i]<=begin_real[i] or begin_out[i]>=end_real[i]:
                        no_overlap_num += 1
                    else:
                        overlap_num += 1
                        if begin_out[i]<=begin_real[i] and end_out[i]>=end_real[i]:
                            out_overlap_num += 1
                        elif begin_out[i]>begin_real[i] and end_out[i]<end_real[i]:
                            in_overlap_num += 1
                        else:
                            part_overlap_num += 1
                pass
            print("test: %d/%d, task: %s, D loss: %f"% (num, len(test_seq), task_name[seq], loss.item()), end="\r")

        print("test: %d/%d, task: %s, loss: %f,task: %s, loss: %f, acc: %f, task: %s, loss: %f"% (epoch, total_epoch, \
                                                    task_name[0], loss_Denoise_total/(len(test_ultra)/batch_size), \
                                                    task_name[2], loss_Discriminator_total/(len(test_cl_ultra)/batch_size), right_num/len(test_cl_ultra),\
                                                    task_name[3], loss_Regression_total/(len(test_rg_ultra)/batch_size),))
        print("FP Rate:%.4f"% (FP_num/N_num))
 
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

        torch.save({
            'epoch': epoch,
            'model_Enc_state_dict': Encoder.state_dict(),
            'model_Res_state_dict': ResNet.state_dict(),
            'model_Den_state_dict': Denoise.state_dict(),
            'model_Dis_state_dict': Discriminator.state_dict(),
            'model_Reg_state_dict': Regression.state_dict(),

            'optimizer_1_state_dict': opt1.state_dict(),
            'optimizer_2_state_dict': opt2.state_dict(),
            'optimizer_3_state_dict': opt3.state_dict(),

            'scheduler_1_state_dict': scheduler1.state_dict(),
            'scheduler_2_state_dict': scheduler2.state_dict(),
            'scheduler_3_state_dict': scheduler3.state_dict(),
            }, model_save_path)
            






    
