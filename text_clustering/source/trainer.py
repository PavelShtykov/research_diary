import pandas as pd
import os, sys, pickle
import torch.nn as nn
import torch
import numpy as np
from model import ModelBERTrain
from dataset import Dataset
import logging
from logging import StreamHandler, Formatter
from itertools import product

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='### [%(asctime)s: %(name)s, %(levelname)s] %(message)s'))
logger.addHandler(handler)


class Trainer:
    def __init__(self, grid):
        self.grid = grid

    @staticmethod
    def preproc_param_grid(curr_params_grid):
        return [
            {k[0]: k[1] for k in zip(curr_params_grid.keys(), vals)}
            for vals in product(*curr_params_grid.values())
        ]

    def run(self):
        for curr_name in self.grid['names']:
            logger.info(f'Start ds: {curr_name}')

            ds = Dataset(curr_name)
            X_train, y_train = ds.get_X_y('train')
            X_test, y_test = ds.get_X_y('test')

            for curr_model, curr_params_grid in self.grid['models'].items():
                logger.info(f'Start model: {curr_model}')

                for curr_param in self.preproc_param_grid(curr_params_grid):
                    logger.info(f'Start config: {curr_param}')

                    if curr_model.lower() == 'bertopic':
                        model = ModelBERTrain(name_of_ds=curr_name, **curr_param)
                    else:
                        raise NotImplementedError
                    
                    prod_train = ds.get_produced(model, 'train', True)
                    prod_test = ds.get_produced(model, 'test', True)

                    if prod_train.shape[0] != 0:
                        logger.warning('ALREADY TRAINED!!!!!!!') 

                    res_train = model.fit_predict(X_train)
                    model.save()

                    if len(res_train) == 3:
                        prod_train = pd.DataFrame(
                            {
                                'clusters': res_train[0][0],
                                'embs': [aa for aa in res_train[1]],
                                'reduce_embs': [aa for aa in res_train[2]]
                            }
                        )
                        ds.set_produced(prod_train, model, 'train')
                    else:
                        raise NotImplementedError
                    
                    res_test = model.predict(X_test)
                    if len(res_test) == 3:
                        prod_test = pd.DataFrame(
                            {
                                'clusters': res_test[0][0],
                                'embs': [aa for aa in res_test[1]],
                                'reduce_embs': [aa for aa in res_test[2]]
                            }
                        )
                        ds.set_produced(prod_test, model, 'test')
                    else:
                        raise NotImplementedError
                    

                    # TODO!!! stats

