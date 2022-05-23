"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
from sklearn import datasets
from sklearn import neighbors
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']
        
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


# """ 
#     NeighborsDataset
#     Returns an image with one of its neighbors.
# """
# class NeighborsDataset(Dataset):
#     def __init__(self, dataset, indices, num_neighbors=None):
#         super(NeighborsDataset, self).__init__()
#         transform = dataset.transform
        
#         if isinstance(transform, dict):
#             self.anchor_transform = transform['standard']
#             self.neighbor_transform = transform['augment']
#         else:
#             self.anchor_transform = transform
#             self.neighbor_transform = transform
       
#         dataset.transform = None
#         self.dataset = dataset
#         self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
#         if num_neighbors is not None:
#             self.indices = self.indices[:, :num_neighbors+1]
#         assert(self.indices.shape[0] == len(self.dataset))

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         output = {}
#         anchor = self.dataset.__getitem__(index)
        
#         neighbor_index = np.random.choice(self.indices[index], 1)[0]
#         neighbor = self.dataset.__getitem__(neighbor_index)

#         anchor['image'] = self.anchor_transform(anchor['image'])
#         neighbor['image'] = self.neighbor_transform(neighbor['image'])

#         output['anchor'] = anchor['image']
#         output['neighbor'] = neighbor['image'] 
#         output['possible_neighbors'] = torch.from_numpy(self.indices[index])
#         output['target'] = anchor['target']
        
#         return output

class NeighborsDataset(Dataset):
    def __init__(self, embds: np.ndarray, pos_neighs=5):
        self.embs_dataset = embds
        nn = NearestNeighbors(n_neighbors=pos_neighs, n_jobs=1)
        nn.fit(self.embs_dataset)
        self.neighbors_idx = torch.from_numpy(nn.kneighbors(self.embs_dataset)[1])
        self.embs_dataset = torch.from_numpy(self.embs_dataset)


    def __len__(self):
        return self.embs_dataset.__len__()
    
    def __getitem__(self, idx):
        output = {}
        anchor = self.embs_dataset[idx]

        negbrs_idxs = self.neighbors_idx[idx]
        negbr_idx = np.random.choice(negbrs_idxs, size=1)
        negbr = self.embs_dataset[negbr_idx[0]]

        output["anchor"] = anchor
        output["neighbor"] = negbr
        output["possible_neighbors"] = negbrs_idxs
        output["target"] = torch.tensor([0], dtype=torch.int16)

            
        return output



