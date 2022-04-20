<h1 align="center"> Pytorch ReID </h1>
<h2 align="center"> Strong, Small, Friendly </h2>

## Students
|Nguyễn Thái Bình| Bùi Văn Hưng| Đặng Lê Minh|
|----------------|-------------|-------------|
|19127103|19127414|19127044|

- **Strong.** It is consistent with the new baseline result in several top-conference works, e.g., [Joint Discriminative and Generative Learning for Person Re-identification(CVPR19)](https://arxiv.org/abs/1904.07223), [Beyond Part Models: Person Retrieval with Refined Part Pooling(ECCV18)](https://arxiv.org/abs/1711.09349), [Camera Style Adaptation for Person Re-identification(CVPR18)](https://arxiv.org/abs/1711.10295). We arrived Rank@1=88.24%, mAP=70.68% only with softmax loss. 

- **Small.** With fp16 (supported by Nvidia apex), our baseline could be trained with only 2GB GPU memory.

- **Friendly.** You may use the off-the-shelf options to apply many state-of-the-art tricks in one line.
Besides, if you are new to object re-ID

![](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/show.png)
                                  sample result

## Table of contents
* [Trained Model](#trained-model)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset--preparation)
    * [Train](#train)
    * [Test](#test)
    * [Evaluation](#evaluation)


Hyperparameters and architectures, that were used to generate the result. 
Some of them (i.e. learning rate) are far from optimal. Do not hesitate to change them and see the effect. 

P.S. With similar structure, we arrived **Rank@1=87.74% mAP=69.46%** with [Matconvnet](http://www.vlfeat.org/matconvnet/). (batchsize=8, dropout=0.75) 
You may refer to [Here](https://github.com/layumi/Person_reID_baseline_matconvnet).
Different framework need to be tuned in a different way.

   
## Trained Model
I re-trained several models, and the results may be different with the original one. Just for a quick reference, you may directly use these models. 
The download link is [Here](https://drive.google.com/open?id=1XVEYb0TN2SbBYOqf8SzazfYZlpH9CxyE).

|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
| [EfficientNet-b4] | 85.78% | 66.80% |  `python train.py --use_efficient --name eff; python test.py --name eff` |
| [ResNet-50 (fp16)] | 88.03% | 71.40% | `python train.py --name fp16 --fp16 --train_all` |
| [ResNet-50] | 88.84% | 71.59% |  `python train.py --train_all` |
| [ResNet-50-ibn] | 89.13% | 73.40% | `python train.py --train_all --name res-ibn --ibn` |


### Losses 
   
|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
| CE | 92.01% | 79.31% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_100 --total 100 ; python test.py  --name  warm5_s1_b32_lr8_p0.5_100`|
| CE + Sphere [[Paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf) | 92.01% | 79.39% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_sphere100 --sphere --total 100; python test.py --name warm5_s1_b32_lr8_p0.5_sphere100` |
| CE + Triplet [[Paper]](https://arxiv.org/pdf/1703.07737) | 92.40%	| 79.71% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_triplet100 --triplet --total 100; python test.py  --name warm5_s1_b32_lr8_p0.5_triplet100` |


### Model Structure
You may learn more from `model.py`. 
We add one linear layer(bottleneck), one batchnorm layer and relu.

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+
- timm `pip install timm` for Swin-Transformer with Pytorch >1.7.0
- pretrainedmodels via `pip install pretrainedmodels`
- [Optional] apex (for float16) 
- [Optional] [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)

**(Some reports found that updating numpy can arrive the right accuracy. If you only get 50~80 Top1 Accuracy, just try it.)**
We have successfully run the code based on numpy 1.12.1 and 1.13.1 .

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- [Optinal] You may skip it. Install apex from the source
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0/0.4.0/0.5.0/1.0.0 and Torchvision 0.2.0/0.2.1 .

### Dataset & Preparation
Download [Market1501 Dataset](http://www.liangzheng.com.cn/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)

Preparation: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remember to change the dataset path to your own path.

Futhermore, you also can test our code on [DukeMTMC-reID Dataset]( [GoogleDriver](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O) or ([BaiduYun](https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw) password: bhbh)).
Our baseline code is not such high on DukeMTMC-reID **Rank@1=64.23%, mAP=43.92%**. Hyperparameters are need to be tuned.

- [Optional] [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) is a generated pedestrian dataset of 128,307 images for training a robust model.

### Train
Train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Train a model with random erasing by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path --erasing_p 0.5
```

### Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--batchsize` batch size.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


### Evaluation
```bash
python evaluate.py
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.
You may also try `evaluate_gpu.py` to conduct a faster evaluation with GPU.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation (consistent with the Market1501 original code).

### re-ranking
```bash
python evaluate_rerank.py
```
**It may take more than 10G Memory to run.** So run it on a powerful machine if possible. 

It will output Rank@1, Rank@5, Rank@10 and mAP results.
```

Basic Model
```bib
@article{zheng2018discriminatively,
  title={A discriminatively learned CNN embedding for person reidentification},
  author={Zheng, Zhedong and Zheng, Liang and Yang, Yi},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  volume={14},
  number={1},
  pages={13},
  year={2018},
  publisher={ACM}
}

@article{zheng2020vehiclenet,
  title={VehicleNet: Learning Robust Visual Representation for Vehicle Re-identification},
  author={Zheng, Zhedong and Ruan, Tao and Wei, Yunchao and Yang, Yi and Mei, Tao},
  journal={IEEE Transaction on Multimedia (TMM)},
  year={2020}
}
```

