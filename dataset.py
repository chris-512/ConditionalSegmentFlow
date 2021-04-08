import os
import cv2
import glob
import json

import numpy as np
from torch.utils.data import Dataset

# labelmap
labelmap_dict = {
    'background': 0,
    'aircon': 1,
    'bed': 2,
    'bookshelf': 3,
    'couch': 4,
    'refrigerator': 5,
    'table': 6,
    'tv': 7
}

number_of_labels = len(labelmap_dict.keys()) - 1

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
def decode_obj(file_path, coeff_x=1.0, coeff_y=1.0):

    with open(file_path) as f:
        anno_dict = json.load(f)

        class_label = labelmap_dict[anno_dict['label']]
        rand_index = np.random.choice(
            len(anno_dict['depth_sample_point_estim']), 1)[0]
        sample_pt_y, sample_pt_x = anno_dict['depth_sample_point_estim'][rand_index]

    sample_pt = np.array((sample_pt_y, sample_pt_x, class_label))

    object = np.expand_dims(np.expand_dims(np.expand_dims(
        sample_pt, 0), 0), 3).astype(np.float32)
    return object


class DataLoader():
    def __init__(self, root):
        self.root = root
        self.rgbs = []
        self.annos = []
        self.load_image_paths()
        self.load_anno_paths()

    def load_image_paths(self):
        self.rgbs.extend(
            sorted(glob.glob(os.path.join(self.root, '*.crop.jpg'))))

        print("Number of images: ", len(self.rgbs))

    def load_anno_paths(self):
        self.annos.extend(
            sorted(glob.glob(os.path.join(self.root, '*.crop.json'))))
        print("Number of annotations: ", len(self.annos))


class SamplePointData(Dataset):
    def __init__(self, width=320, height=576, split='train', test_id=0, root=None):

        self.split = split
        self.width = width
        self.height = height
        # train: <data_dir>/train
        # test: <data_dir>/test
        self.dataset = DataLoader(os.path.join(root, split))
        self.test_id = test_id

    def __len__(self):
        if self.split == 'train':
            return len(self.dataset.rgbs)
        else:
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

        gt_object = decode_obj(anno_path)
        color_img = decode_img(img_path, width=self.width, height=self.height)
        one_hot_mask = get_mask(self.width,
                                self.height, class_id=int(gt_object[0, 0, 2, 0]))  # class id

        input = np.squeeze(np.concatenate([color_img, one_hot_mask], axis=1))
        output = gt_object[0, 0, 0: 2, 0]  # (y, x)

        if self.split == 'train':
            s = np.random.uniform(0, 1)
            if s > 0.5:
                input = np.flip(input, 2).copy()  # horizontally
                output[1] = 1.0 - output[1]
            s = np.random.uniform(0, 1)
            if s > 0.5:
                input = np.flip(input, 1).copy()  # vertically
                output[0] = 1.0 - output[0]
        return input, output


def get_mask(width, height, class_id=1):
    input = np.tile(np.expand_dims(np.expand_dims(
        np.eye(number_of_labels)[class_id-1], axis=-1), axis=-1), (1, height, width))
    input = np.expand_dims(input, axis=0)
    # input = np.ones([height, width]) * (fill_value + 1)
    # input = np.expand_dims(np.expand_dims(input, axis=0), axis=0)
    return input


if __name__ == '__main__':
    import torch

    dataset = SamplePointData(width=256, height=256,
                              split='train', root='../data')

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=True)

    for x, y in train_loader:
        import pdb
        pdb.set_trace()
