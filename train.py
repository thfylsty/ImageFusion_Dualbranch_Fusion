# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from densefuse_net import DenseFuseNet
from ssim import SSIM 
from utils import mkdir,AEDataset,gradient,hist_similar
import os
import time
from loss_network import LossNetwork
# import scipy.io as scio
# import numpy as np
# from matplotlib import pyplot as plt

save_name = "H_"
os.chdir(r'./')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Parameters

root_ir = r'../../../../../Disk_B/KAIST-RGBIR/visible'
root_vi = r'../../../../../Disk_B/KAIST-RGBIR/lwir'
root_ir_TNO = r'../../data/ir'
root_vi_TNO = r'../../data/vi'
root_ir = root_ir_TNO
root_vi = root_vi_TNO
# root_val = r'D:\document\Study\data\VOC2007\val'
train_path = './train_result/'
epochs = 500
batch_size = 64
print("epochs",epochs,"batch_size",batch_size)
device = 'cuda'
lr = 1e-3
lambd = 1


# Dataset
data_k = AEDataset(root_ir,root_vi, resize=128, gray = True)
loader_k = DataLoader(data_k, batch_size = batch_size, shuffle=True)
# data_vi = AEDataset(root_vi, resize= 128, gray = True)
# loader_vi = DataLoader(data_vi, batch_size = batch_size, shuffle=True)
data_TNO = AEDataset(root_ir_TNO,root_vi_TNO, resize= 128, gray = True)
loader_TNO = DataLoader(data_TNO, batch_size = batch_size, shuffle=True)

img_num = (len(data_TNO)+len(data_k))/2
print("Load ---- {} pairs of images: KAIST:[{}] + TNO:[{}]".format(img_num,len(data_k)/2,len(data_TNO)/2))
# print("Load ---- {} pairs of images:TNO:[{}]".format(img_num,len(data_TNO)))
# data_val = AEDataset(root_val, resize= [256,256], transform = None, gray = True)
# loader_val = DataLoader(data_val, batch_size = 100, shuffle=True)

# Model
model = DenseFuseNet().to(device)
# checkpoint = torch.load('./train_result/H_model_weight_new.pkl')
# model.load_state_dict(checkpoint['weight'])
print(model)
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
    verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

# mode(str)- 模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。
# factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor
# patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
# verbose(bool)- 是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))
# threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
# 当 threshold_mode == rel，并且 mode == max 时， dynamic_threshold = best * ( 1 +threshold )；
# 当 threshold_mode == rel，并且 mode == min 时， dynamic_threshold = best * ( 1 -threshold )；
# 当 threshold_mode == abs，并且 mode== max 时， dynamic_threshold = best + threshold ；
# 当 threshold_mode == abs，并且 mode == max 时， dynamic_threshold = best - threshold；
# threshold(float)- 配合 threshold_mode 使用。
# cooldown(int)- “冷却时间“，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
# min_lr(float or list)- 学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
# eps(float)- 学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。

MSE_fun = nn.MSELoss()
SSIM_fun = SSIM()
CrossEntropyLoss = nn.CrossEntropyLoss()
with torch.no_grad():
    loss_network = LossNetwork()
    loss_network.to(device)
loss_network.eval()


# Training
mse_train = []
ssim_train = []
loss_train = []
mse_val = []
ssim_val = []
loss_val = []
gradient_train = []
mkdir(train_path)
min_loss = 100
print('============ Training Begins [epochs:{}] ==============='.format(epochs))
steps = len(loader_TNO)+len(loader_k)
steps = steps/2
KASIST_num = len(loader_k)
s_time = time.time()
loss = torch.zeros(1)

grd_loss_all = []
hist_loss_all = []
mse_loss_all = []
perceptual_loss_all = []

for iteration in range(epochs):

    scheduler.step(loss.item())
    img_name = "KASIST"
    imgs_T = iter(loader_TNO)
    imgs_K = iter(loader_k)
    # img_irs = iter(loader_ir)
    # img_vis_TNO = iter(loader_vi_TNO)
    # img_irs_TNO = iter(loader_ir_TNO)
    tqdms = tqdm(range(int(steps)))
    for index in tqdms:

        if index == KASIST_num:
            img_name = "TNO"
        # img = next(imgs).to(device)
        if index < KASIST_num:
            img = next(imgs_K).to(device)
        else:
            # print(2)
            img = next(imgs_T).to(device)
        optimizer.zero_grad()

        img_re= model(img)

        mse_loss = MSE_fun(img,img_re)
        grd_loss = MSE_fun(gradient(img), gradient(img_re))
        hist_loss = hist_similar(img, img_re.detach()) * 0.001
        # std_loss = torch.abs(img_re.std() - img.std())
        std_loss = hist_loss

        # 感知损失
        with torch.no_grad():
            x = img.detach()
        features = loss_network(x)
        features_re = loss_network(img_re)

        with torch.no_grad():
            f_x_vi1 = features[1].detach()
            f_x_vi2 = features[2].detach()
            f_x_ir3 = features[3].detach()
            f_x_ir4 = features[4].detach()

        perceptual_loss = MSE_fun(features_re[1], f_x_vi1)+MSE_fun(features_re[2], f_x_vi2) + \
                         MSE_fun(features_re[3], f_x_ir3)+MSE_fun(features_re[4], f_x_ir4)

        std_loss = std_loss
        perceptual_loss = perceptual_loss*1000

        grd_loss_all.append(grd_loss.item())
        hist_loss_all.append(hist_loss.item())
        mse_loss_all.append(mse_loss.item())
        perceptual_loss_all.append(perceptual_loss.item())

        loss = mse_loss +grd_loss + std_loss+ perceptual_loss
        loss.backward()
        optimizer.step()

        e_time = time.time()-s_time
        last_time = epochs*int(steps)*(e_time/(iteration*int(steps)+index+1))-e_time

        tqdms.set_description('%d MSGP[%.5f %.5f %.5f %.5f] T[%d:%d:%d] lr:%.4f '%
          (iteration,mse_loss.item(),std_loss.item(),grd_loss.item(),perceptual_loss.item(),last_time/3600,last_time/60%60,
           last_time%60,optimizer.param_groups[0]['lr']*1000))
        # scheduler.step(loss.item())



        # print('[%d,%d] -   Train    - MSE: %.10f, SSIM: %.10f '%
        #   (iteration,index,mse_loss.item(),ssim_loss.item()))
        # if iteration>1:
        #     mse_train.append(mse_loss.item())
        #     ssim_train.append(ssim_loss.item())
        #     loss_train.append(loss.item())
        #     gradient_train.append(grd_loss.item())

            
        # with torch.no_grad():
        #     tmp1, tmp2 = .0, .0
        #     for _, img in enumerate(loader_val):
        #         img = img.to(device)
        #         img_recon = model(img)
        #         tmp1 += (MSE_fun(img,img_recon)*img.shape[0]).item()
        #         tmp2 += (SSIM_fun(img,img_recon)*img.shape[0]).item()
        #     tmp3 = tmp1+lambd*tmp2
        #     mse_val.append(tmp1/data_val.__len__())
        #     ssim_val.append(tmp1/data_val.__len__())
        #     loss_val.append(tmp1/data_val.__len__())
        # print('[%d,%d] - Validation - MSE: %.10f, SSIM: %.10f'%
        #   (iteration,index,mse_val[-1],ssim_val[-1]))
        # scio.savemat(os.path.join(train_path, 'TrainData.mat'),
        #              {'mse_train': np.array(mse_train),
        #               'ssim_train': np.array(ssim_train),
        #               'loss_train': np.array(loss_train)})
        # scio.savemat(os.path.join(train_path, 'ValData.mat'),
        #              {'mse_val': np.array(mse_val),
        #               'ssim_val': np.array(ssim_val),
        #               'loss_val': np.array(loss_val)})

    # plt.figure(figsize=[12,8])
    # plt.subplot(2,2,1), plt.semilogy(mse_train), plt.title('mse train')
    # plt.subplot(2,2,2), plt.semilogy(ssim_train), plt.title('ssim train')
    # plt.subplot(2,2,3), plt.semilogy(gradient_train), plt.title('grd val')
    # plt.subplot(2,2,4), plt.semilogy(loss_train), plt.title('loss train')
    # plt.subplot(2,2,4), plt.semilogy(mse_val), plt.title('mse val')
    # plt.subplot(2,3,5), plt.semilogy(ssim_val), plt.title('ssim val')
    # plt.subplot(2,3,6), plt.semilogy(loss_val), plt.title('loss val')


    # plt.savefig(os.path.join(train_path,'curve.png'),dpi=90)

    if min_loss > loss.item():
        min_loss = loss.item()
        torch.save({'weight': model.state_dict(), 'epoch': iteration, 'batch_index': index},
                   os.path.join(train_path, save_name+'best.pkl'))
        print('[%d] - Best model is saved -' % (iteration))


    if (iteration+1) % 10 ==0 and iteration != 0:
        torch.save( {'weight': model.state_dict(), 'epoch':iteration, 'batch_index': index},
                   os.path.join(train_path,save_name+'model_weight_new.pkl'))
        print('[%d] - model is saved -'%(iteration))

import scipy.io as scio

scio.savemat('./train_result/loss_all.mat',{'grd_loss_all':grd_loss_all,'hist_loss_all':hist_loss_all,
                                        'mse_loss_all':mse_loss_all,'perceptual_loss_all':perceptual_loss_all})
