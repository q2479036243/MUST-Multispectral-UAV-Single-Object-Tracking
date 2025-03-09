import os
import cv2
import jpeg4py
import numpy as np
from shutil import copyfile

npy_path= '/data/users/qinhaolin01/MUST-BIT/datasets/MUST-BIT/HSIData'
save_path = '/data/users/qinhaolin01/MUST-BIT/datasets/MUST-BIT/HSICV'
splits = ['test', 'train']

for split in splits:
    npy_seqs_path = os.path.join(npy_path, split)
    save_seqs_path = os.path.join(save_path, split)
    if not os.path.exists(save_seqs_path):
        os.makedirs(save_seqs_path)
    for npy_seq in os.listdir(npy_seqs_path):
        if npy_seq.endswith('txt'):
            list_path = os.path.join(npy_seqs_path, npy_seq)
            list_save_path = os.path.join(save_seqs_path, npy_seq)
            copyfile(list_path, list_save_path)
        else:
            npy_seq_path = os.path.join(npy_seqs_path, npy_seq)
            save_seq_path = os.path.join(save_seqs_path, npy_seq)
            if not os.path.exists(save_seq_path):
                os.makedirs(save_seq_path)
            for npy_file in os.listdir(npy_seq_path):
                if npy_file.endswith('txt'):
                    txt_path = os.path.join(npy_seq_path, npy_file)
                    txt_save_path = os.path.join(save_seq_path, npy_file)
                    copyfile(txt_path, txt_save_path)
                else:
                    npy_data = np.load(os.path.join(npy_seq_path, npy_file))
                    im1 = cv2.imwrite(os.path.join(save_seq_path, npy_file.replace('.npy','_img1.jpg')), npy_data[:,:,0:3])
                    im2 = cv2.imwrite(os.path.join(save_seq_path, npy_file.replace('.npy','_img2.jpg')), npy_data[:,:,3:6])
                    im3 = cv2.imwrite(os.path.join(save_seq_path, npy_file.replace('.npy','_img3.jpg')), npy_data[:,:,5:8])