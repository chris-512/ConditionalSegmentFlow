import os
import cv2
import glob
import json

import numpy as np
from torch.utils.data import Dataset

# read an image amd resize when needed


def decode_img(file_path, width=None, height=None):

    img = cv2.imread(file_path)

    img = img / 255.0
    img = np.subtract(img, 0.4)
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height),
                         interpolation=cv2.INTER_LANCZOS4)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    return img

# read the float file containing the object information


def decode_sample_points(file_path, nr_samples_from_mask=30):

    segim = cv2.imread(file_path)
    h, w, c = segim.shape
    cls_ids = np.unique(segim)
    cls_ids = cls_ids[cls_ids != 255]
    cls_ids = cls_ids[cls_ids == 0]

    if len(cls_ids) != 0:
        # chosen_cls_id = np.random.choice(cls_ids)
        nr_positives = nr_samples_from_mask
        index_yx = np.where(segim == 0)
        sample_index = np.random.choice(len(index_yx[0]), nr_positives)
        positives = np.transpose(np.asarray([index_yx[0][sample_index] / float(h),
                                             index_yx[1][sample_index] / float(w)]), (1, 0))

        nr_negatives = 2 * nr_samples_from_mask
        index_yx = np.where(segim != 0)
        sample_index = np.random.choice(
            len(index_yx[0]), nr_negatives)  # twice the number of postive samples

        negatives = np.transpose(np.asarray([index_yx[0][sample_index] / float(h),
                                             index_yx[1][sample_index] / float(w)]), (1, 0))

        sample_pts = np.vstack([positives, negatives])
        gt_label = 1
    else:
        sample_pts = np.vstack([np.asarray([np.random.uniform(0, 1), np.random.uniform(0, 1)])
                                for i in range(3 * nr_samples_from_mask)])
        gt_label = 0

    gt_sample_points = sample_pts.astype(np.float32)
    return gt_sample_points, gt_label


class DataLoader():
    def __init__(self, root, split=None):
        self.root = root
        self.rgbs = []
        self.annos = []
        self.split = split
        if self.split == 'train2017':
            self.sample_no = 5000
        else:
            self.sample_no = 200
        self.load_image_paths()
        self.load_anno_paths()

    def load_image_paths(self):
        self.rgbs.extend(
            sorted(glob.glob(os.path.join(self.root, 'images', self.split, '*.jpg')))[:self.sample_no])
        print("Number of images: ", len(self.rgbs))

    def load_anno_paths(self):
        self.annos.extend(
            sorted(glob.glob(os.path.join(self.root, 'annotations', self.split, '*.png')))[:self.sample_no])
        print("Number of annotations: ", len(self.annos))


class SamplePointData(Dataset):
    def __init__(self, split='train2017', width=320, height=576, test_id=0, root=None):
        self.split = split
        self.width = width
        self.height = height
        # train: <data_dir>/train
        # test: <data_dir>/test

        labeltxt = os.path.join(root, '../labels.txt')
        self.labelmap = {}
        self.labelmap[0] = "unlabeled"
        with open(labeltxt) as f:
            for line in f.readlines():
                id_, label_ = line.split(':')
                id_ = int(id_)
                if id_ == 0 or id_ > 1:
                    continue
                # only animals (16-25, bird, cat, dog ~~~ giraffe)
                print(id_)
                self.labelmap[id_] = label_.strip()

        self.class_size = len(self.labelmap)

        self.dataset = DataLoader(root, split=self.split)
        self.test_id = test_id

    def __len__(self):
        return len(self.dataset.rgbs)

    def __getitem__(self, idx):
        if self.split == 'train':
            rand_index = np.random.choice(len(self.dataset.rgbs), 1)[0]
            img_path = self.dataset.rgbs[rand_index]
            anno_path = self.dataset.annos[rand_index]
        else:
            rand_index = np.random.choice(len(self.dataset.rgbs), 1)[0]
            img_path = self.dataset.rgbs[rand_index]
            anno_path = self.dataset.annos[rand_index]

        gt_sample_points, gt_label = decode_sample_points(anno_path)
        color_img = decode_img(img_path, width=self.width, height=self.height)
        one_hot_mask = get_onehot_tensor(self.class_size, self.width,
                                         self.height, gt_label)  # class id

        input = np.squeeze(np.concatenate([color_img, one_hot_mask], axis=1))
        output = gt_sample_points

        if self.split == 'train':
            s = np.random.uniform(0, 1)
            if s > 0.5:
                input = np.flip(input, 2).copy()  # horizontally
                output[1] = 1.0 - output[1]
            s = np.random.uniform(0, 1)
            """
            if s > 0.5:
                input = np.flip(input, 1).copy()  # vertically
                output[0] = 1.0 - output[0]
            """
        return input, output


def get_onehot_tensor(class_size, width, height, class_id):
    input = np.tile(np.expand_dims(np.expand_dims(
        np.eye(class_size)[class_id], axis=-1), axis=-1), (1, height, width))
    input = np.expand_dims(input, axis=0)
    return input


if __name__ == '__main__':

    import torch

    dataset = SamplePointData(width=256, height=256,
                              root='/home/syk/cocostuff/dataset')

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=True)

    for x, y in train_loader:
        import pdb
        pdb.set_trace()
