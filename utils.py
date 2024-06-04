
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
import numpy as np
from dgl.data import citation_graph as citegrh
import pandas as pd
import h5py
import os
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def save_predictions(output_dir, node_names, predictions):
    with open(os.path.join(output_dir, 'predictions.tsv'), 'w') as f:
           f.write('ID\tName\tProb_pos\n')
           for pred_idx in range(predictions.shape[0]):
               f.write('{}\t{}\t{}\n'.format(node_names[pred_idx, 0],
                                                node_names[pred_idx, 1],
                                                predictions[pred_idx, 0])
                )


def load_hdf_data(path, network_name='network', feature_name='features'):
    """Load a GCN input HDF5 container and return its content.

    This funtion reads an already preprocessed data set containing all the
    data needed for training a GCN model in a medical application.
    It extracts a network, features for all of the nodes, the names of the
    nodes (genes) and training, testing and validation splits.

    Parameters:
    ---------
    path:               Path to the container
    network_name:       Sometimes, there might be different networks in the
                        same HDF5 container. This name specifies one of those.
                        Default is: 'network'
    feature_name:       The name of the features of the nodes. Default is: 'features'

    Returns:
    A tuple with all of the data in the order: network, features, y_train, y_val,
    y_test, train_mask, val_mask, test_mask, node names.
    """
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        node_names = f['gene_names'][:]
        y_train = f['y_train'][:]
        y_test = f['y_test'][:]
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None
        train_mask = f['mask_train'][:]
        test_mask = f['mask_test'][:]
        if 'mask_val' in f:
            val_mask = f['mask_val'][:]
        else:
            val_mask = None
        if 'feature_names' in f:
            feature_names = f['feature_names'][:]
        else:
            feature_names = None
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names