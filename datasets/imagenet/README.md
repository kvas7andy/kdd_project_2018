# ImageNet dataset

Please provide code in '$../../src/imagenet.py', which automatically downloads 
$N_k \approx 10$ images for each of 1000 class in 'dataset/imagenet/images' folder. 
At the same time produce text file eval.txt with filenames, its locations 
"relative to the eval.txt location" and class label (from 0 to 1000)

**Important**: 'eval.txt' and 'train.txt' formats:

'''

image1_path  label1

image2_path  label2

image3_path  label3

...

'''
### Two options:


#### 1. Best one: right code for imagenet files aggregation
1. Download images from urls according to labels/classes (WNID) and predefined parameter $$N_K \approx 10$$ - number of random instances for each of 1000 classes from ImagNet
2. Construct eval.txt file 
3. Use PyTorch code to produce centers (will be in imagenet.py)


Go on with that [tutorial](http://fungai.org/2017/12/12/download-imagenet-images-by-wnid/), hope it will help
Still need to change the ImageNet_utils to get not all of the images but some of them:

[classes wnid_label_name mapping](https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt)

[wnid_urls](http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz))

From my point of view, the idea is:
1. Find file with {WNID: image_url} list (I found official xml file [here (~1Gb)](http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz))
2. Download only several files. We need just $N_k$ files, so read url, get an image and after getting $N_k$ for class k (for the same WNID)- stop downloading 
3. Use pandas to store counters and file locations
4. Label imagenames according to the index of WNID, which corresponds to this image_url
5. (Opt.) need to associate unique WNID with unique class name from this [structure_realeased.xml](http://www.image-net.org/api/xml/structure_released.xml) file and number them according to this [dictionaty from smbd gist](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
6. Provide eval.txt file with images, their filenames and class labels (store in rows with commas)

### 1. Worst one: download whole dataset
1. Download whole dataset
2. Use the whole data to produce centers with PyTorch code



