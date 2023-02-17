"""
MIT License

Copyright (c) 2018 Sagar Vinodababu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
import os

import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class UnitDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/datasets.py
    """

    def __init__(self, data_folder, data_name, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        """
        assert split in {'TRAIN', 'VAL', 'TEST'}

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, split + '_CAPTIONS_' + data_name + '.json')) as j:
            captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, split + '_CAPLENS_' + data_name + '.json')) as j:
            self.caplens = json.load(j)

        # Load word map
        with open(os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')) as j:
            word_map = json.load(j)
        
        self.captions = torch.LongTensor(captions)
        self.padding_masks = self.captions == word_map["<pad>"]
        assert isinstance(self.padding_masks, torch.BoolTensor)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        return self.captions[i], self.padding_masks[i], self.caplens[i]

    def __len__(self):
        return self.dataset_size


class UnitImageDataset(UnitDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/datasets.py
    """

    def __init__(self, data_folder, data_name, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        """
        super().__init__(data_folder, data_name, split)

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        img = self.transform(img)
        return self.captions[i], self.padding_masks[i], self.caplens[i], img