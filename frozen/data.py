import os
import os.path as osp
import json
import logging


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import GPT2Tokenizer, AutoFeatureExtractor
from pytorch_lightning import LightningDataModule


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_TOKEN = "<image>"
SPECIAL_TOKEN_DICT = {'additional_special_tokens': [IMAGE_TOKEN]}
NUM_IMAGE_TOKENS = 2
PAD_TOKEN_ID = 1


class COCODataset(Dataset):
    def __init__(
        self,
        name='COCO',
        path=None,
        split='val',
        year=2017,
        image_transform=None,
        tokenizer=None,
        num_image_tokens=0,
    ):
        super().__init__()
        assert split in ('train', 'val')
        assert year in (2014, 2017)
        logging.warn(f'num_image_tokens = {num_image_tokens}')

        self.path = osp.abspath(osp.expanduser(path))
        self.split = split
        self.year = year
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens

        if self.tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-2.7b")
        
        if not IMAGE_TOKEN in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens(SPECIAL_TOKEN_DICT)
        
        self.setup_dataset()

    def setup_dataset(self):
        split, year = self.split, self.year
        self.split_name = f'{split}{year}'
        self.image_dir = osp.join(self.path, 'images', self.split_name)
        self.annotation_file = osp.join(
            self.path, 'annotations', f'captions_{self.split_name}.json')

        with open(self.annotation_file, 'r') as f:
            json_data = json.load(f)
            annotations = json_data['annotations']
        
        image_dict = dict()
        for item in json_data['images']:
            image_dict[item['id']] = item

        self.annotations = annotations
        self.image_dict = image_dict        

    def __len__(self):
        return len(self.annotations)

    def _read_image(self, index):
        image_id = self.annotations[index]['image_id']
        file_name = self.image_dict[image_id]['file_name']
        file_path = osp.join(self.image_dir, file_name)
        image = Image.open(file_path)
        if self.image_transform is not None:
            image = self.image_transform(image, return_tensors='pt')
        return image_id, image['pixel_values']

    def _add_image_tokens(self, caption):
        N = self.num_image_tokens
        if N is not None or N > 0:
            tokens = ' '.join([IMAGE_TOKEN for x in range(N)])
            caption = f'{tokens} {caption}'
        return caption

    def __getitem__(self, index):
        image_id, image = self._read_image(index)
        caption = self.annotations[index]['caption']
        caption = self._add_image_tokens(caption)
        inputs = self.tokenizer(caption, return_tensors='pt')

        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        image_token_mask = inputs['input_ids'] == image_token_id

        return {
            'pixel_values': image,
            'caption': caption,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'image_token_mask': image_token_mask.long(),
            'item_id': index,
            'image_id': image_id,
            'caption_id': self.annotations[index]['id'],
        }



class CC3MDataset(COCODataset):
    def setup_dataset(self):
        assert self.split in ('train', 'val')
        script_path = osp.join(self.path, 'script/DownloadConceptualCaptions/')
        annotation_file = 'Train_GCC-training.tsv'
        if self.split == 'val':
            annotation_file = 'Validation_GCC-1.1.0-Validation.tsv'
        annotations = pd.read_csv(
            osp.join(script_path, annotation_file),
            sep='\t',
            names=['caption', 'image_url'],
        )

        split_ = 'training' if self.split == 'train' else 'validation'
        download_file = osp.join(script_path, f'downloaded_{split_}_report.tsv')
        download_report = pd.read_csv(
            osp.join(script_path, download_file),
            sep='\t',
            names=["image_url", "split", "status", "file_name"])
        download_report = download_report[download_report['status'] == 200]
        download_report = download_report.loc[:, ['image_url', 'file_name']]
        records = annotations.merge(download_report)
        self.annotations = records.to_dict('records')
        self.image_dir = osp.join(self.path, split_)

    def _read_image(self, index):
        file_name = self.annotations[index]['file_name']
        file_path = osp.join(self.path, file_name)
        try:
            image = Image.open(file_path)
            image = image.convert('RGB') if image.mode != 'RGB' else image
        except UnidentifiedImageError:
            image = None
        
        if image is not None and self.image_transform is not None:
            image = self.image_transform(image, return_tensors='pt')
        elif image is None:
            image = {'pixel_values': None}
        return image['pixel_values']

    
    def __getitem__(self, index):
        item = self.annotations[index]
        image = self._read_image(index)
        caption = item['caption']
        caption = self._add_image_tokens(caption)
        inputs = self.tokenizer(caption, return_tensors='pt')
        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        image_token_mask = inputs['input_ids'] == image_token_id

        return {
            'pixel_values': image,
            'caption': caption,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'image_token_mask': image_token_mask.long(),
            'item_id': index,
            'image_id': -1,
            'caption_id': -1,
        }


class CaptioningDataModule(LightningDataModule):
    def __init__(
        self,
        config=dict(),
    ):
        super().__init__()
        self.config = config
        self.init_tokenizer()
        self.init_image_transform()
        self.load_splits()

    @property
    def loader_config(self):
        default_config = {
            'num_workers': 0,
            'pin_memory': False,
            'batch_size': 16,
        }
        return self.config.get('loader', default_config)
    
    @property
    def dataset_config(self):
        return self.config.get('dataset', dict())

    @property
    def model_config(self):
        return self.config.get('model', dict())

    def init_tokenizer(self):
        arch = self.model_config.get('text_encoder', 'facebook/opt-2.7b')
        self.tokenizer = GPT2Tokenizer.from_pretrained(arch)

    def init_image_transform(self):
        arch = self.model_config.get('image_encoder', 'microsoft/resnet-50')
        self.image_transform = AutoFeatureExtractor.from_pretrained(arch)

    def load_splits(self):
        self.train_data = self.load_split('train')
        self.val_data = self.load_split('val')

    def load_split(self, split):
        N = self.model_config.get('num_image_tokens', NUM_IMAGE_TOKENS)
        dataset = self.dataset_config.get('name', 'COCO')
        if dataset == 'COCO':
            dataset_class = COCODataset
        elif dataset == 'CC3M':
            dataset_class = CC3MDataset

        return dataset_class(
            split=split,
            tokenizer=self.tokenizer,
            image_transform=self.image_transform,
            **self.dataset_config,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            collate_fn=collate_fn,
            shuffle=True,
            **self.loader_config,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            collate_fn=collate_fn,
            shuffle=False,
            **self.loader_config
        )

    def predict_dataloader(self):
        return self.val_dataloader()

    
def collate_fn(batch):
    batch = [x for x in batch if x['pixel_values'] is not None]
    batch_size = len(batch)
    longest = max([x['input_ids'].numel() for x in batch])
    pixel_values = torch.cat([x['pixel_values'] for x in batch])

    def init_helper(value, dtype):
        array = torch.empty((batch_size, longest), dtype=dtype)
        array.fill_(value)
        return array

    input_ids = init_helper(PAD_TOKEN_ID, torch.long)
    attention_mask = init_helper(0, torch.long)
    image_token_mask = init_helper(False, torch.long)

    for i in range(batch_size):
        length = batch[i]['input_ids'].numel()
        input_ids[i, :length] = batch[i]['input_ids']
        attention_mask[i, :length] = batch[i]['attention_mask']
        image_token_mask[i, :length] = batch[i]['image_token_mask']

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image_token_mask': image_token_mask,
        'item_ids': [x['item_id'] for x in batch],
        'captions': [x['caption'] for x in batch],
        'image_ids': [x['image_id'] for x in batch],
        'caption_ids': [x['caption_id'] for x in batch],
    }
    
