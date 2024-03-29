import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class UnitDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    Only load units for training a pure unit-LM
    """

    def __init__(self, data_folder, data_name, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL'
        """
        self.data_folder = data_folder
        self.data_name = data_name
        self.split = split
        assert self.split in {'TRAIN', 'VAL'}

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)
    
    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        return caption, caplen

    def __len__(self):
        return self.dataset_size

class UnitDatasetMask(UnitDataset):
    def __init__(self, data_folder, data_name, split, word_map):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        super().__init__(data_folder, data_name, split)

        self.word_map = word_map
        
        self.captions = torch.LongTensor(self.captions)

        '''
            In training,
            as we input: "<start>, 1, 2, ... , t"
            to decoder:  "1, 2, ... , t, <end>"
            We should better not input "<end>" to decoder, 
            because decoding actually stopped, and decoding "<end>" doesn't mean anything
        '''
        # self.padding_masks = self.captions == self.word_map["<pad>"]
        padding_masks = self.captions == self.word_map["<pad>"]
        endding_masks = self.captions == self.word_map["<end>"]
        self.padding_masks = padding_masks != endding_masks
        assert isinstance(self.padding_masks, torch.BoolTensor)
    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        padding_mask = self.padding_masks[i]

        return caption, caplen, padding_mask


class CaptionDataset(UnitDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        super().__init__(data_folder, data_name, split)

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

class CaptionDataset_transformer(CaptionDataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        super().__init__(data_folder, data_name, split, transform)
        # NOTE: 之前下面这种写法(改了)
        # super().__init__(data_folder, data_name, split, transform=None)
        # 导致了送入的参数从来不 transform
        with open(os.path.join(self.data_folder, 'WORDMAP_' + self.data_name + '.json')) as j:
            self.word_map = json.load(j)
        
        self.captions = torch.LongTensor(self.captions)

        '''
            In training,
            as we input: "<start>, 1, 2, ... , t"
            to decoder:  "1, 2, ... , t, <end>"
            We should better not input "<end>" to decoder, 
            because decoding actually stopped, and decoding "<end>" doesn't mean anything
        '''
        # self.padding_masks = self.captions == self.word_map["<pad>"]
        padding_masks = self.captions == self.word_map["<pad>"]
        endding_masks = self.captions == self.word_map["<end>"]
        self.padding_masks = padding_masks != endding_masks
        assert isinstance(self.padding_masks, torch.BoolTensor)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        padding_mask = self.padding_masks[i]

        if self.split == 'TRAIN':
            return img, caption, caplen, padding_mask
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            all_padding_mask = torch.BoolTensor(
                self.padding_masks[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, padding_mask, all_captions, all_padding_mask
