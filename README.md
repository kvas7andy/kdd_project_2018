# KDD Project

## General info

This project replicates the paper "Cost-Effective Training of Deep CNNs with Active Model Adaptation" [[KDD2018 paper page]](https://www.kdd.org/kdd2018/accepted-papers/view/cost-effective-training-of-deep-cnns-with-active-model-adaptation)

#### Structure

├── datasets              contains files for ImageNet, DogVsCat datasets
│   ├── c_x_A_B           stores calculated features from layers VGG, AlexNet models (from pretrained models) 
│   │   ├── alex
│   │   └── vgg16
│   ├── dogvscat          
│   │   ├── test1
│   │   └── train
│   └── imagenet          Files related to structured ImageNet dataset from the 
│       └── train
├── libs                  src files for ImageNet dataset download
├── src                   all source files for the project, except ImageNet downloaders

#### Prerequirements:
-python3.6
-pytorch
-numpy
-scipy
-pandas
-pillow
-urllib

## Compilation
No compilation necessary  but preliminary datasets download is necessary, go to folders of [ImageNet](datasets/imagenet) and [DogVSCat](datasets/dogvscat) dataset

## Execution

python3.6 main.py
