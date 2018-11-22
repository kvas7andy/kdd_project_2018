# KDD Project

## General info

This project replicates the paper "Cost-Effective Training of Deep CNNs with Active Model Adaptation" [[KDD2018 paper page]](https://www.kdd.org/kdd2018/accepted-papers/view/cost-effective-training-of-deep-cnns-with-active-model-adaptation)

#### Structure

| File/Folder          | Description           | 
| ------------- |:-------------:| 
| ├── datasets       | Contains files for ImageNet, DogVsCat datasets (source files as well as images and .txt contents) | 
| /├── c_x_A_B      | Stores calculated features from layers VGG, AlexNet models (from pretrained models)       |   
|  //├── alex  |      |
|  //└── vgg16  |      |
|  /├── dogvscat   |      |
|  //├── test1  |      |
|  //└── train  |      |
|  /└── imagenet  | Files related to structured ImageNet dataset from the      |
|  ├── libs    | Src files for ImageNet dataset download     | 
|  ├── src   |  All source files for the project, except ImageNet downloaders   | 

#### Prerequirements:
* python3.6
* pytorch
* numpy
* scipy
* pandas
* pillow
* urllib

Tested on Ubuntu 16.04 with 12 Gb GPU

## Compilation
No compilation necessary  but preliminary datasets download is necessary, go to folders of [ImageNet](datasets/imagenet) and [DogVSCat](datasets/dogvscat) dataset

## Execution

0. Download data: specific to datasets. ImageNet download with [downloadutils.py](downloadutils.py) and target datasets, for instance, from files. See README.md respectively for [ImageNet](datasets/dogvscat/README.md) and [DogVSCat](datasets/imagenet/README.md).

1. Firstly change parameters in [utils.py](src/utils.py):

* Sampling related
  * `active_samepl_size` (mini batch to check the score composed of distinctiveness and uncertainty)
  * `active_batch_size` (for the exact labeled batch of images to ask for)
  * `eval_batch_size` (after how many images to make evaluation)
* Model related
  * `model_type` - Deep CNNs to use ([Alex](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) or [VGG16](https://arxiv.org/abs/1409.1556) available)
  * `active` - whether apply active batching
  * `lamb` - speed parameter for disctictiveness/uncertainty trade-off
  * `transform` - whether to use data augmentation
  * `alternate` - whether to prevent using images from one class consequently for transfer learning with 1 image, only for miaty class (project team improvement proposal)
* For results report:
  * `eval` - whether perform evaluation on datasets, and save accuracy of the model into the [src/historyactive_...](src/historyactive_...) file after 100 batches
  * `eval_every` - evaluation frequency (number of acative batches)
  * `report_every` - report evaluation performance by printing
  
2. Then execute [src/main.py](src/main.py) with `python3.6 main.py`
