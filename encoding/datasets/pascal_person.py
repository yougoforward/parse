import os
import numpy as np
import random

import torch

from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
from .base import BaseDataset
import torchvision.transforms as transforms

class PersonSegmentation(BaseDataset):
    CLASSES = [
        'background', 'head', 'torso', 'upper_arm', 'lower_arm', 'upper_leg', 
        'lower_leg'
    ]
    NUM_CLASS = 7
    BASE_DIR = 'Person'
    def __init__(self, root=os.path.expanduser('./data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(PersonSegmentation, self).__init__(root, split, mode, transform,
                                              target_transform, **kwargs)
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationPart')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join('./encoding/datasets', 'Pascal')
        if self.split == 'train':
            _split_f = os.path.join(_splits_dir, 'train_id.txt')
        elif self.split == 'val':
            _split_f = os.path.join(_splits_dir, 'val_id.txt')
        elif self.split == 'test':
            _split_f = os.path.join(_splits_dir, 'val_id.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in tqdm(lines):
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if self.mode != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n')+".png")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if self.mode != 'test':
            assert (len(self.images) == len(self.masks))
        
        self.colorjitter = transforms.ColorJitter(brightness=0.1, contrast=0.5, saturation=0.5, hue=0.1)
            

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        target = Image.open(self.masks[index])
            
        # synchrosized transform
        if self.mode == 'train':
            img, target = self._sync_transform( img, target)

        elif self.mode == 'val':
            img, target = self._val_sync_transform( img, target)
        else:
            assert self.mode == 'testval'
            target = self._mask_transform(target)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # Generate target maps
        seg_half = np.array(target)
        seg_half[(seg_half > 0) & (seg_half <= 4)] = 1
        seg_half[(seg_half > 4) & (seg_half < 255)] = 2
        seg_half = torch.from_numpy(seg_half).long()
        seg_full = np.array(target)
        seg_full[(seg_full > 0) & (seg_full < 255)] = 1
        seg_full = torch.from_numpy(seg_full).long()
        return img, target, seg_half, seg_full

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        #colorjitter and rotate
        if random.random() < 0.5:
            # img = Image.fromarray(img)
            # mask = Image.fromarray(mask)
            img = self.colorjitter(img)
            # random rotate -10~10, mask using NN rotate
            deg = random.uniform(-10, 10)
            img = img.rotate(deg, resample=Image.BILINEAR)
            mask = mask.rotate(deg, resample=Image.NEAREST)
        # # random rotate -10~10, mask using NN rotate
        # deg = random.uniform(-10, 10)
        # img = img.rotate(deg, resample=Image.BILINEAR)
        # mask = mask.rotate(deg, resample=Image.NEAREST)
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0
