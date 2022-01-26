import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import random
from PIL import Image


rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width, grayscale=False):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    if grayscale:
        image_decoded = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized



class Reconstruction3DDataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, num_frames=16,
                 img_extension='.jpg', dataset='ped2', jump=[2], hold=[2], return_normal_seq=False):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._num_frames = num_frames

        self.extension = img_extension
        self.dataset = dataset

        self.setup()
        self.samples, self.background_models = self.get_all_samples()

        self.jump = jump
        self.hold = hold
        self.return_normal_seq = return_normal_seq  # for fast and slow moving

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*/'))
        for video in sorted(videos):
            video_name = video.split('/')[-2]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*' + self.extension))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        background_models = []
        videos = glob.glob(os.path.join(self.dir, '*/'))
        for video in sorted(videos):
            video_name = video.split('/')[-2]

            for i in range(len(self.videos[video_name]['frame']) - self._num_frames + 1):
                frames.append(self.videos[video_name]['frame'][i])

        return frames, background_models

    def __getitem__(self, index):
        # index = 8
        video_name = self.samples[index].split('/')[-2]
        if self.dataset == 'shanghai' and 'training' in self.samples[index]:
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2]) - 1
        else:
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        batch = []
        for i in range(self._num_frames):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                                  self._resize_width, grayscale=True)

            if self.transform is not None:
                batch.append(self.transform(image))

        return np.stack(batch, axis=1)

    def __len__(self):
        return len(self.samples)


class Reconstruction3DDataLoaderJump(Reconstruction3DDataLoader):
    def __getitem__(self, index):
        # index = 8
        video_name = self.samples[index].split('/')[-2]
        if self.dataset == 'shanghai' and 'training' in self.samples[index]:  # bcos my shanghai's start from 1
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2]) - 1
        else:
            frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        batch = []
        normal_batch = []
        jump = random.choice(self.jump)

        retry = 0
        while len(self.videos[video_name]['frame']) < frame_name + (self._num_frames-1) * jump and retry < 10:
            # reselect the frame_name
            frame_name = np.random.randint(len(self.videos[video_name]['frame']))
            retry += 1

        for i in range(self._num_frames):
            image = np_load_frame(self.videos[video_name]['frame'][min(frame_name + i*jump, len(self.videos[video_name]['frame'])-1)], self._resize_height,
                                  self._resize_width, grayscale=True)

            if self.transform is not None:
                batch.append(self.transform(image))

        if self.return_normal_seq:
            for i in range(self._num_frames):
                image = np_load_frame(self.videos[video_name]['frame'][min(frame_name + i, len(self.videos[video_name]['frame'])-1)], self._resize_height,
                                      self._resize_width, grayscale=True)

                if self.transform is not None:
                    normal_batch.append(self.transform(image))
            return np.stack(batch, axis=1), np.stack(normal_batch, axis=1)

        else:
            return np.stack(batch, axis=1)

