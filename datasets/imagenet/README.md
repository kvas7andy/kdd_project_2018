# ImageNet dataset

There were no possibility to download the whole ImageNet dataset (error on the [original website](http://image-net.org/download.php))


In order to have control of the images, which are used for centers and in order to compute everything on 1 GPU only we utilized  [ImageNet Utils](https://github.com/tzutalin/ImageNet_Utils) github repository.

#### Requirements:
* python2.7
* urllib

1. In order to download N images for each of 1000 classes of ImageNet use file in root folder [../../downloadutils.py](../../downloadutils.py) and specify the `n_image_per_class` parameter (default 10).
2. Execute `python2.7 downloadutils.py`
3. After this all images will be stored in the `train` folder and the information about path, class, id and url of the each image
4. Execute [../../imagenet_file_utils.py](../../imagenet_file_utils.py) file to generate `train.txt` (with filtering out possible dublicate downloads)

After producing `train.txt` this file will be used by the [../../src/main.py](../../src/main.py) to geneate `train_centers.txt` file particularly with the information of centers for each class, stating path and class of the images (1000 lines of all).



