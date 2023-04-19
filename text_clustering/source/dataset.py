import pandas as pd
import os, sys, pickle
import torch.nn as nn
import torch
import numpy as np

import logging
from logging import StreamHandler, Formatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='### [%(asctime)s: %(name)s, %(levelname)s] %(message)s'))
logger.addHandler(handler)

class CoreDataset:
    def __init__(self, path, mode='r'):
        self._path = path
        self.mode = mode
        self.rewrite = None
        self._df = None
        self.read()

    def read(self):
        if os.path.isfile(self._path):
            if os.path.splitext(self._path)[1] == '.csv':
                self._df = pd.read_csv(self._path)
            elif os.path.splitext(self._path)[1] == '.parquet':
                self._df = pd.read_parquet(self._path)
            else:
                raise ValueError
        else: 
            self._df = None 
    
    def write(self):
        if self.mode == 'r':
            raise TypeError
        if self._df is None:
            raise ValueError
        
        if os.path.splitext(self._path)[1] == '.csv':
            self._df.to_csv(self._path, index=False)
        elif os.path.splitext(self._path)[1] == '.parquet':
            self._df.to_parquet(self._path, index=False)
        else:
            raise ValueError

    def get_df(self):
        if self._df is None:
            raise ValueError

        return self._df.copy()
    
    def update_df(self, new_df):
        if self.mode == 'r':
            raise TypeError
        
        if (self._df is None 
            or set(self._df.columns).issubset(set(new_df.columns))):
            self._df = new_df


class Dataset:
    def __init__(self, name):
        self.base_path = os.path.join('./datasets', name)

        self.all_ds = CoreDataset(os.path.join(self.base_path, 'all.csv'), 'r')
        self.train_ds = CoreDataset(os.path.join(self.base_path, 'train.csv'), 'r')
        self.test_ds = CoreDataset(os.path.join(self.base_path, 'test.csv'), 'r')
        self.rewrite = None

        self._dir_struct_validate()
        self.readed_prod = {}

    def _dir_struct_validate(self):
        def make_dir(name):
            if not os.path.isdir(os.path.join(self.base_path, name)):
                os.mkdir(os.path.join(self.base_path, name))
            
        make_dir('produced')

    def get_X(self, part='train'):
        if part == 'all':
            return self.all_ds.get_df()['text'].to_list()
        elif part == 'train':
            return self.train_ds.get_df()['text'].to_list()
        elif part == 'test':
            return self.test_ds.get_df()['text'].to_list()
        else:
            raise ValueError
        
    def get_short_X(self, part='train'):
        if part == 'all':
            return self.all_ds.get_df()['short_text'].to_list()
        elif part == 'train':
            return self.train_ds.get_df()['short_text'].to_list()
        elif part == 'test':
            return self.test_ds.get_df()['short_text'].to_list()
        else:
            raise ValueError
        
    def get_y(self, part='train'):
        if part == 'all':
            return self.all_ds.get_df()['int_label'].values
        elif part == 'train':
            return self.train_ds.get_df()['int_label'].values
        elif part == 'test':
            return self.test_ds.get_df()['int_label'].values
        else:
            raise ValueError
        
    def get_X_y(self, part='train'):
        return self.get_X(part), self.get_y(part)
    
    def get_short_X_y(self, part='train'):
        return self.get_short_X(part), self.get_y(part)

    def get_text_y(self, part='train'):
        if part == 'all':
            return self.all_ds.get_df()['label'].to_list()
        elif part == 'train':
            return self.train_ds.get_df()['label'].to_list()
        elif part == 'test':
            return self.test_ds.get_df()['label'].to_list()
        else:
            raise ValueError
        
    @staticmethod
    def _name_produced(model, part):
        return f'{part}_{model.get_hash()}.parquet'

    def get_produced(self, model, part='train', rewrite=False):
        name = self._name_produced(model, part)
        path = os.path.join(self.base_path, 'produced', name)

        if not os.path.isfile(path):
            rewrite = True
        if self.rewrite is None:
            self.rewrite = rewrite

        if name in self.readed_prod.keys():
            return self.readed_prod[name].get_df()
        else:
            self.readed_prod[name] = CoreDataset(path, 'w')
            self.readed_prod[name].update_df(pd.DataFrame(columns=['embs', 'clusters']))

            return self.readed_prod[name].get_df()
        
    def set_produced(self, new_df, model, part='train'):
        if self.rewrite == False:
            raise ValueError
        
        if new_df.shape[0] != self.get_y(part).shape[0]:
            raise IndexError

        name = self._name_produced(model, part)

        if name not in self.readed_prod.keys():
            raise ValueError
        
        self.readed_prod[name].update_df(new_df)
        self.readed_prod[name].write()
