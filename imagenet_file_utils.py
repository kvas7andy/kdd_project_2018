import os
import re
from os import path as pth
import time

import pandas as pd

max_id = 10

db_dir = 'datasets/imagenet'
if False:
    l = sorted(os.listdir(pth.join(db_dir, "train")))
    #parser = lambda x: re.findall(r"[\w']+",x.replace('_', '.'))[:-1]#x.replace('_', ' ').replace('.', ' ').split('\W+')
    # print(list(zip(list(map(parser, l)))))
    # img_name class number url
    s = [[x] + re.findall(r"[\w']+", x.replace('_', '.'))[:-1] for x in l if '.txt' not in x]
    with open(pth.join(db_dir, "2601_images_urls.txt"), "r") as fr:
        preimaged_urls= fr.readlines()
        preimaged_urls = [[x.split(' ')[-1][:-1], x.split(' ')[1]] for x in preimaged_urls if 'Downloading' in x]
        preimaged_db = pd.DataFrame(preimaged_urls, columns = ['img_name', 'url'])


    db = pd.DataFrame(s, columns=['img_name', 'class', 'img_id'])
    db = db.join(preimaged_db.set_index('img_name'), on='img_name')
    joinpath = lambda x: pth.join('..', db_dir, "train", x)
    db[["class", "img_id"]] = db[["class", "img_id"]].apply(pd.to_numeric)
    db = db.sort_values(['class', 'img_id'], ascending=[True, True])
    db["img_name"] = db["img_name"].map(joinpath)
    db.to_csv(pth.join(db_dir, "train", "train_whole.txt"), sep=' ', index=False, header=False)

if True:
    preimaged_db = pd.read_csv(pth.join(db_dir, "train", "train_whole.txt"), sep=' ', header=None,
                               names=['img_name', 'class', 'img_id', 'url'],
                               dtype={'img_name':str, 'class': int,
                                      'img_id': int, 'url':str})
    #preimaged_db["img_name"] = preimaged_db["img_name"].apply(lambda x: '/'.join(x.split('/')[-2:]))
    preimaged_db = preimaged_db.sort_values(['class', 'img_id'], ascending=[True, True])
    preimaged_db.drop_duplicates(subset=['class', 'img_id'], keep='first', inplace=True)
    preimaged_db.to_csv(pth.join(db_dir, 'train', "train_whole.txt"), sep=' ', index=False, header=False)
    db = preimaged_db[preimaged_db["img_id"] < max_id][['img_name', 'class']]
    db.to_csv(pth.join(db_dir, "train.txt"), sep=' ', index=False, header=False)
