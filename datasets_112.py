import os
import numpy as np
import cv2
import pandas as pd
import random
import mxnet
from mxnet.gluon.data.vision import transforms as mx_transforms


from PIL import Image, ImageFilter

import utils

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

transforms_op = mx_transforms.Compose([mx_transforms.Resize(120),
                                    mx_transforms.RandomResizedCrop(112, scale=(0.94,1.0), ratio=(1.0,1.0)),
                                    mx_transforms.ToTensor(),
                                    mx_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class DataBatch():

    def __init__(self, data, label, contlabel):
        self.batchdata_ = data
        self.batchlabel_ = label
        self.batchcontlabel_ = contlabel
    '''
    @property
    def provide_data(self):
        return [("data", self.batchdata_[0].shape)]

    @property
    def provide_label(self):
        return [("label", self.batchlabel_[0].shape)]

    @property
    def provide_contlabel(self):
        return [("contlabel", self.batchcontlabel_[0].shape)]
    '''


class Custom_iter():

     def __init__(self, data_dir, filenamelist_path, batch_size = 16, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
         self.img_ext_ = img_ext
         self.annot_ext_ = annot_ext
         self.image_mode_ = image_mode

         self.filename_list_ = get_list_from_filenames(filenamelist_path)

         self.batch_size_ = batch_size
         self.length_ = len(self.filename_list_)
         self.num_batches_ = self.length_ / self.batch_size_
         self.data_dir_ = data_dir

         self.initialization()


     def initialization(self):
        self.cur_batch_ = 0
        random.shuffle(self.filename_list_)

     def __iter__(self):
         return self

     def next(self):
         if self.cur_batch_ < self.num_batches_:
             index = self.cur_batch_*self.batch_size_
             batch_imgs, batch_labels, batch_contlabels = self.get_batch_imgs_labels(index)
             self.cur_batch_ += 1
             return DataBatch(batch_imgs, batch_labels, batch_contlabels)
         else:
             self.initialization()
             raise StopIteration

     def get_batch_imgs_labels(self, index):

         for i in range(0, self.batch_size_):
             img, label, contlabel = self.get_img_label(index + i)
             if i == 0:
                 img_batch = img.copy()
                 label_batch = label.copy()
                 contlabel_batch = contlabel.copy()
             else:
                 label_batch = np.vstack((label_batch, label))
                 contlabel_batch = np.vstack((contlabel_batch, contlabel))
                 img_batch = np.vstack((img_batch, img))
         return img_batch, label_batch, contlabel_batch

     def get_img_label(self, index):

         img = Image.open(os.path.join(self.data_dir_, self.filename_list_[index] + self.img_ext_))
         img = img.convert(self.image_mode_)
         mat_path = os.path.join(self.data_dir_, self.filename_list_[index] + self.annot_ext_)

         # Crop the face loosely
         pt2d = utils.get_pt2d_from_mat(mat_path)
         x_min = min(pt2d[0, :])
         y_min = min(pt2d[1, :])
         x_max = max(pt2d[0, :])
         y_max = max(pt2d[1, :])
         # k = 0.2 to 0.40
         k = np.random.random_sample() * 0.2 + 0.2
         x_min -= 0.6 * k * abs(x_max - x_min)
         y_min -= 2 * k * abs(y_max - y_min)
         x_max += 0.6 * k * abs(x_max - x_min)
         y_max += 0.6 * k * abs(y_max - y_min)
         img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

         # We get the pose in radians
         pose = utils.get_ypr_from_mat(mat_path)
         # And convert to degrees.
         pitch = pose[0] * 180 / np.pi
         yaw = pose[1] * 180 / np.pi
         roll = pose[2] * 180 / np.pi

         # Flip
         rnd = np.random.random_sample()
         if rnd < 0.5:
             yaw = -yaw
             roll = -roll
             img = img.transpose(Image.FLIP_LEFT_RIGHT)

         # Blur
         rnd = np.random.random_sample()
         if rnd < 0.05:
             img = img.filter(ImageFilter.BLUR)

         # Bin values
         bins = np.array(range(-99, 102, 3))
         binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

         # Get target tensors
         label = binned_pose
         cont_label = np.array([yaw, pitch, roll])

         img_new = transforms_op(mxnet.nd.array(img))
         img_numpy = img_new.asnumpy()

         return np.expand_dims(img_numpy, axis=0), np.expand_dims(label, axis=0), np.expand_dims(cont_label, axis=0)

     def __len__(self):
         # 122,450
         return self.length_






