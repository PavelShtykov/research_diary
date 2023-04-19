import pandas as pd
import os, sys, pickle
import torch.nn as nn
import torch
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

import logging
from logging import StreamHandler, Formatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = StreamHandler(stream=sys.stdout)
handler.setFormatter(Formatter(fmt='### [%(asctime)s: %(name)s, %(levelname)s] %(message)s'))
logger.addHandler(handler)

class ModelBase:
    def fit_predict(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError
    
    def load(self):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError
    

class ModelBERTrain(ModelBase):
    def __init__(
            self, 
            name_of_ds,
            sbert_model='paraphrase-multilingual-mpnet-base-v2',
            reducer_name='UMAP',
            reducer_n_neighbors=15,
            reduce_dim=5,
            reducer_metric='cosine',
            clustering_name='HDBSCAN',
            clustering_param: dict = {
                'min_cluster_size':10, 'metric': 'euclidean', 
                'prediction_data': True, 
                'min_samples': 5
            }
        ):
        self.name_of_ds = name_of_ds
        self.hash_str = {
            'sbert_model': sbert_model, 
            'reducer_name': reducer_name, 
            'reduce_dim': reduce_dim, 
            'reducer_n_neighbors': reducer_n_neighbors, 
            'reducer_metric': reducer_metric,
            'clustering_name': clustering_name, 
            **clustering_param
        }
        self.hash_str = '+'.join([f'{k}={v}' for k, v in self.hash_str.items()])

        logger.info(f'Hash of model: {self.hash_str}')

        if clustering_name == 'HDBSCAN':
            self.clustering = HDBSCAN(**clustering_param)
        else: 
            raise ValueError
        
        if reducer_name == 'UMAP':
            self.reducer = UMAP(n_neighbors=reducer_n_neighbors, n_components=reduce_dim, 
                    min_dist=0.0, metric=reducer_metric, random_state=42)
        else:
            raise ValueError
        
        self.model = BERTopic(
            language='english',
            top_n_words=10,
            umap_model=self.reducer,
            hdbscan_model=self.clustering,
            embedding_model=sbert_model
        )

    def get_hash(self):
        return self.hash_str
    
    def fit_predict(self, X, curr_embs=None, curr_reduce_embs=None):
        res = self.model.fit_transform(X, embeddings=curr_embs, reduce_embeddings=curr_reduce_embs)
        red_embs = self.model.result_reduce_embs
        embs = self.model.result_embs

        return res, embs, red_embs
    
    def predict(self, X, curr_embs=None, curr_reduce_embs=None):
        res = self.model.transform(X, embeddings=curr_embs, reduce_embeddings=curr_reduce_embs)
        red_embs = self.model.result_reduce_embs
        embs = self.model.result_embs

        return res, embs, red_embs


    def save(self):
        path = os.path.join('./models', 'bertopic', self.name_of_ds)
        os.makedirs(path, exist_ok=True)
        
        path = os.path.join(path, f'{self.hash_str}.model')
        self.model.save(path)

        logger.info(f'Save model {self.hash_str}')

    def load(self, manual_path=None):
        if manual_path is not None:
            path = manual_path
        else:
            path = os.path.join('./models', 'bertopic', self.name_of_ds, f'{self.hash_str}.model')

        if not os.path.isfile(path):
            raise ValueError
        
        self.model.load(path)

        logger.info(f'Load model {self.hash_str}')
