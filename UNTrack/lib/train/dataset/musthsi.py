import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import hsijpg_loader
from lib.train.admin import env_settings


class MUSTHSI(BaseVideoDataset):
    def __init__(self, root=None, image_loader=hsijpg_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - 数据集路径
            image_loader - 数据读取方式,因为是npy,因此用must_loader
            split - train和test分别对应两个不同的文件夹
            seq_ids - 可以设置id号来选择使用文件夹内指定序列,默认为None,表示使用全部序列
            data_fraction - 一个小于1的数,例如0.8,表示从全部序列中随机选择80%使用
        """
        root = env_settings().musthsi_dir if root is None else root
        super().__init__('MUSTHSI', root, image_loader)

        # all folders inside the root
        self.root = os.path.join(self.root, split)
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the root path
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'musthsi'

    def has_class_info(self):
        return True
    
    def has_out_of_view_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list
    
    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path.replace('HSIJPG','FalseColor'), "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte()
        
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        with open(os.path.join(seq_path.replace('HSIJPG','FalseColor'), "full_occlusion.txt")) as f:
            name_list = list(csv.reader(f))
        name_list = [name[0] for name in name_list]
        
        name_path = os.path.join(seq_path, name_list[frame_id] + '_img1.jpg')    # frames start from 1
        
        return name_path

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        class_name = seq_path.split('/')[-1].split('-')[1]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            
        object_meta = OrderedDict({'object_class_name': class_name,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
