#!/usr/bin/env python 

import os
import glob
import json
import cv2
from lxml import etree

def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.

  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.

  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def point_annotate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('y: ', y)
        print('x: ', x)
        print('h: ', img_dim['h'])
        print('w: ', img_dim['w'])
        print(y / img_dim['h'])
        print(x / img_dim['w'])

        single_anno['depth_sample_point_estim'].append((y / img_dim['h'], x / img_dim['w']))

cv2.namedWindow('bbox')
cv2.setMouseCallback('bbox', point_annotate)

img_dim = {}
single_anno = {}
single_anno['depth_sample_point_estim'] = []
single_anno['label'] = None

jpeg_files = glob.glob('*.jpg')
crop_jpeg_files = glob.glob('*.crop.jpg')
jpeg_files = list(set(jpeg_files) - set(crop_jpeg_files))
jpeg_files = sorted(jpeg_files)
bbox_anno_files = glob.glob(os.path.join('/home/mlsyn91/data/wmb+sr_home/Annotations', '*.xml'))
bbox_anno_files = sorted(bbox_anno_files)

for jpeg_file, bbox_anno in zip(jpeg_files, bbox_anno_files):
    img = cv2.imread(jpeg_file)
    with open(bbox_anno) as f:
        xml_str = f.read()
    xml = etree.fromstring(xml_str)
    xml_dict = recursive_parse_xml_to_dict(xml)
    print(xml_dict.keys())
    
    print('image shape: ', img.shape)

    cv2.imshow('image', img)

    for i, obj_dict in enumerate(xml_dict['annotation']['object']):
        class_name = obj_dict['name']
        bnd_box = obj_dict['bndbox']
        #print(bnd_box['xmin'], bnd_box['ymin'], bnd_box['xmax'], bnd_box['ymax'])
        xmin = int(float(bnd_box['xmin']))
        xmin = 0 if (xmin < 0) else xmin
        ymin = int(float(bnd_box['ymin']))
        ymin = 0 if (ymin < 0) else ymin 
        xmax = int(float(bnd_box['xmax']))
        xmax = img.shape[1]-1 if (xmax > img.shape[1]-1) else xmax
        ymax = int(float(bnd_box['ymax']))
        ymax = img.shape[0]-1 if (ymax > img.shape[0]-1) else ymax

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        print(xmin, ymin, xmax, ymax)
        print('class label: ', class_name)
        single_anno['label'] = class_name
        bbox_img = img[ymin:ymax+1, xmin:xmax+1, :]
        img_dim['h'] = bbox_img.shape[0]
        img_dim['w'] = bbox_img.shape[1]
        cv2.imshow('bbox', bbox_img)
        key = cv2.waitKey(0)
        if key == ord('s'):
            name, ext = os.path.splitext(jpeg_file)
            crop_img_file = name + '_%d.crop.jpg' % i
            anno_file = name + '_%d.crop.json' % i
            print('save crop image to %s' % crop_img_file)
            cv2.imwrite(crop_img_file, bbox_img)

            print('save annotation to %s' % anno_file)
            with open(anno_file, 'w+') as f:
                json.dump(single_anno, f)
    
            single_anno['depth_sample_point_estim'] = []
            single_anno['label'] = None
