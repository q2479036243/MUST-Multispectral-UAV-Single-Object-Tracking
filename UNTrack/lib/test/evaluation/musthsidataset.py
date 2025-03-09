import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class MUSTHSIDataset(BaseDataset):
    """ 
    MUST-HSI dataset.
    """
    def __init__(self, split):
        super().__init__()
        # split = 'train'
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test':
            self.base_path = os.path.join(self.env_settings.musthsi_path, split)
        else:
            self.base_path = os.path.join(self.env_settings.musthsi_path, 'train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path.replace('HSIJPG','FalseColor'), sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith("_img1.jpg")]
        frame_list.sort()
        frames_list = [[os.path.join(frames_path, frame), os.path.join(frames_path, frame).replace('img1','img2'), os.path.join(frames_path, frame).replace('img1','img3')] for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'musthsi', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        attrs = ['Similar 0bject', 'Camera Motion', 'Low Resolution', 
                'Partial Occlusion', 'Background Clutter', 'Similar Color', 
                'Similar 0bject', 'Motion Blur', 'Fast Motion',
                'Out of View', 'Illumination Variation', 'Full Occlusion']
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()
        return sequence_list
