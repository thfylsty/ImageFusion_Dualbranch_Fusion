<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">DenseFuse (Pytorch)</h3>


---

<p align="center"> An pytorch implement of DenseFuse.
    <br> 
</p>

## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>
This is a pytorch implement of DenseFuse proposed by this paper, 
[H. Li, X. J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Trans. Image Process., vol. 28, no. 5, pp. 2614–2623, May. 2019.](https://arxiv.org/abs/1804.08361)

The code is writted with torch 1.1.0 and pytorch-ssim.


## 🎈 Usage <a name="usage"></a>

### Quick start 
1. Clone this repo and unpack it. 
2. Download [test dataset](https://github.com/hli1221/imagefusion_densefuse/tree/master/images) and put test images in './images/IV_images'
3. run 'main.py'

### Training
A pretrained model is available in './train_result/model_weight.pkl'. We train it on MS-COCO 2014. There are 82783 images in total, where 1000 images (COCO_train2014_000000574951 ~ COCO_train2014_000000581921.jpg) are used for validation and the rest are for training. In the training phase, all images are resize to 256x256 and are transformed to gray pictures. Model is optimized by Adam with learning rate being 1e-4. The batch size and epoch number are 2 and 4, respectively. Loss function is MSE+lambda x SSIM, where lambda=1. The experiments were implemented with 2080ti GPU and 32GB RAM. It took about 2 hours. 

If you want to re-train this net, you should download MS-COCO 2014 and run 'train.py'


