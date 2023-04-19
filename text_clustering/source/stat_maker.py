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


class StatMaker:
    def __init__(self, grid):
        self.grid = grid

    def count_ds_stats(self):