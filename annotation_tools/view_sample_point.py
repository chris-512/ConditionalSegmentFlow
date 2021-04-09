#!/usr/bin/env python 

import json
import glob
import cv2
import os
import numpy as np

jpeg_files = glob.glob('*.crop.jpg')
jpeg_files = sorted(jpeg_files)
anno_files = glob.glob('*.json')
anno_files = sorted(anno_files)
for jpeg_file, anno_file in zip(jpeg_files, anno_files):
    img = cv2.imread(jpeg_file)


    print('anno file: ', anno_file)
    with open(anno_file) as f:
        anno_dict = json.load(f)
    print(anno_dict)

    hor_flip = False
    ver_flip = False
    s = np.random.uniform(0, 1)
    if s > 0.5:
        img = np.flip(img, 1)
        hor_flip = True
    if s > 0.5:
        img = np.flip(img, 0)
        ver_flip = True

    height = img.shape[0]
    width = img.shape[1]

    img = cv2.UMat(img)
    sample_pts = anno_dict['depth_sample_point_estim']
    for y, x in sample_pts:
        y = int(height * y)
        x = int(width * x)
        if hor_flip:
            x = width - x
        if ver_flip:
            y = height - y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    
    print('label: ', anno_dict['label'])

    cv2.imshow('test', img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
