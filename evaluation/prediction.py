import numpy as np
import torch
from PIL import Image as pimg

__all__ = ['StorePreds', 'StoreSubmissionPreds']


class StorePreds:
    def __init__(self, store_dir, to_img, to_color, remap=None):
        self.store_dir = store_dir
        self.to_img = to_img
        self.to_color = to_color
        self.remap = remap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __str__(self):
        return ''

    def __call__(self, pred, batch, additional):
        self.store_preds(pred, batch)  # store colored predictions
        # self.store_trainid(pred, batch)  # store trainId labels

    def store_preds(self, pred, batch):
        b = self.to_img(batch)
        for p, im, gt, name, subset in zip(pred, b['image'], b['original_labels'], b['name'], b['subset']):
            # TODO: Fix the leftImg8bit image being distorted
            # store_img = np.concatenate([i.astype(np.uint8) for i in [im, self.to_color(p), gt]], axis=0)
            store_img = np.concatenate([i.astype(np.uint8) for i in [self.to_color(p), gt]], axis=0)
            store_img = pimg.fromarray(store_img)
            # store_img.thumbnail((960, 1344))
            # store_img.thumbnail((640, 896))
            store_img.save(f'{self.store_dir}/{subset}/{name}.jpg')
    
    def store_trainid(self, pred, batch):
        b = self.to_img(batch)
        for p, name in zip(pred, b['name']):  # overwriting the original labels with the predictions
            p = p.to(torch.int)
            store_img = self.remap(p)
            store_img = pimg.fromarray(store_img)
            name = name.replace('leftImg8bit', 'gtFine_labelIds')
            store_img.save(f'{self.store_dir}/{name}.png')


class StoreSubmissionPreds:
    def __init__(self, store_dir, remap, to_color=None, store_dir_color=None):
        self.store_dir = store_dir
        self.store_dir_color = store_dir_color
        self.to_color = to_color
        self.remap = remap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __str__(self):
        return ''

    def __call__(self, pred, batch, additional):
        for p, name in zip(pred.astype(np.uint8), batch['name']):
            pimg.fromarray(self.remap(p)).save(f'{self.store_dir}/{name}.png')
            pimg.fromarray(self.to_color(p)).save(f'{self.store_dir_color}/{name}.png')