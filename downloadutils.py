#!/usr/bin/env python
import argparse
import sys
import os
import _init_paths
from libs import imagedownloader
from libs import pref_utils

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Help the user to download, crop, and handle images from ImageNet')
    p.add_argument('--downloadDataset', help='should download dataset', action='store_true', default=True)
    p.add_argument('--wnid', nargs='+', help='ImageNet Wnid. E.g. : n02710324')
    p.add_argument('--downloadImages', help='Should download images', action='store_true', default=False)
    p.add_argument('--downloadOriginalImages', help='Should download original images', action='store_true', default=False)
    p.add_argument('--downloadBoundingBox', help='Should download bouding box annotation files', action='store_true', default=False)
    # p.add_argument('--jobs', '-j', type=int, default=1, help='Number of parallel threads to download')
    # p.add_argument('--timeout', '-t', type=int, default=10, help='Timeout per image in seconds')
    # p.add_argument('--retry', '-r', type=int, default=10, help='Max count of retry for each image')
    p.add_argument('--verbose', '-v', action='store_true', help='Enable verbose log')
    p.add_argument('--wnid_url_map_dir', '-wumd', type=str,
                   default='/home/akvasov/Documents/COMP5331/kdd_project/cost-effective-transfer-learning-master/datasets/imagenet',
                   help='dataset, where to store')

    args = p.parse_args()
    #if args.wnid is None:
    #    print 'No wnid'
    #    sys.exit()


    downloader = imagedownloader.ImageNetDownloader()
    username = None
    accessKey = None
    userInfo = pref_utils.readUserInfo()

    if not userInfo is None:
        username = userInfo[0]
        accessKey = userInfo[1]
    if args.downloadDataset is True:
        file_list = []

        with open(os.path.join(args.wnid_url_map_dir, 'map_clsloc.txt')) as fp:
            #../datasets/imagenet/train/
            #0_1.jpg
            #../datasets/imagenet/train.txt
            #path to the image label

            for line in fp:
                linesegs = line.strip().split(' ')
                label = str(int(linesegs[1])-1)
                label_name = linesegs[2]
                list = downloader.getImageURLsOfWnid(linesegs[0])
                cnt = 0
                for l in list:
                    filename_info = {'dataset_dir':args.wnid_url_map_dir,
                                     'filename':  label + '_' + str(cnt) + '.jpg'}
                    status = downloader.downloadImagesByURLs(linesegs[0], [l],
                                                    filename_info = filename_info)
                    if status is not None:
                        file_list += [filename_info['filename'] + ' ' + label_name]
                        cnt += 1

                    if cnt >= 10:
                        break
        with open(os.path.join(args.wnid_url_map_dir, "train.txt"), "w") as f:
            f.write("\n".join(file_list))


    if args.downloadImages is True:
        for id in args.wnid:
            list = downloader.getImageURLsOfWnid(id)
            list = list[0:19]
            downloader.downloadImagesByURLs(id, list)

    if args.downloadBoundingBox is True:
        for id in args.wnid:
            # Download annotation files
            downloader.downloadBBox(id)

    if args.downloadOriginalImages is True:
    # Download original image, but need to set key and username
        if username is None or accessKey is None:
            username = raw_input('Enter your username : ')
            accessKey = raw_input('Enter your accessKey : ')
            if username and accessKey:
                pref_utils.saveUserInfo(username, accessKey)

        if username is None or accessKey is None:
            print 'need username and accessKey to download original images'
        else:
            for id in args.wnid:
                downloader.downloadOriginalImages(id, username, accessKey)
