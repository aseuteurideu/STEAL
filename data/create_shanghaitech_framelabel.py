import numpy as np
import glob
import os

dataset_root = '../dataset/shanghai/testing'

videos_list = sorted(glob.glob(os.path.join(dataset_root, 'frames', '*')))
anno = None
for video in videos_list:
    anno_file = os.path.join(dataset_root, 'test_frame_mask', os.path.basename(video) + '.npy')

    if anno is None:
        anno = np.load(anno_file)
    else:
        anno = np.concatenate((anno, np.load(anno_file)))

anno = np.expand_dims(anno, 0)
np.save('frame_labels_shanghai.npy', anno)
a = np.load('frame_labels_shanghai.npy')

pass



