# -*- coding: utf-8 -*-
from glob import glob
from densefuse_net import DenseFuseNet
import torch
from PIL import Image
import torchvision.transforms as transforms
import time
import os
from channel_fusion import channel_f as channel_fusion
from utils import mkdir,Strategy
_tensor = transforms.ToTensor()
_pil_gray = transforms.ToPILImage()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = 'cuda'
model = DenseFuseNet().to(device)
checkpoint = torch.load('./train_result/H_best.pkl')
# checkpoint = torch.load('./train_result/model_weight_new.pkl')
model.load_state_dict(checkpoint['weight'])

mkdir("outputs/fea/")
mkdir("outputs/fea/vi/")
mkdir("outputs/fea/ir/")
mkdir("result")
test_ir = './Test_ir/'
test_vi = './Test_vi/'


def load_img(img_path, img_type='gray'):
    img = Image.open(img_path)
    if img_type == 'gray':
        img = img.convert('L')
    return _tensor(img).unsqueeze(0).to(device)

fusename = ['l1','add','channel']

def test(model):
    img_list_ir = glob(test_ir+'*')
    img_num = len(img_list_ir)
    print("Test images num", img_num)
    for i in range(1,img_num+1):
        img1_path = test_ir+str(i)+'.bmp'
        img2_path = test_vi+str(i)+'.bmp'

        img1, img2, = load_img(img1_path), load_img(img2_path)
        s_time = time.time()
        feature1, feature2 = model.encoder(img1,isTest=True),model.encoder(img2,isTest=True)
        for name in fusename:
            if name =='channel':
                features = channel_fusion(feature1, feature2,is_test=True)
                out = model.decoder(features).squeeze(0).detach().cpu()
            else:
                with torch.no_grad():
                    fusion_layer = Strategy(name, 1).to(device)
                    feature_fusion = fusion_layer(feature1, feature2)
                    out = model.decoder(feature_fusion).squeeze(0).detach().cpu()
            e_time = time.time() - s_time
            save_name = 'result/'+name+'/fusion'+str(i)+'.bmp'
            mkdir('result/'+name)
            img_fusion = _pil_gray(out)
            img_fusion.save(save_name)
            print("pic:[%d] %.4fs %s"%(i,e_time,save_name))
with torch.no_grad():
    test(model)
