#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Data available from: https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt

# Using Mouse Cortex Cells dataset
# 3005 cells, 7 distinct cell types to recover

import argparse
import csv

import numpy as np
import pandas as pd
import scanpy as sc

# import os.path
from os import path

import sys
try:
    import anndata as ad
except ImportError as error:
    print (error)
    sys.exit(1)

# need to give full path now when this is imported into Jupyter notebook
#base_dir = '.'
base_dir = '/home/chiaretta/performance/programming/github/GeneVAE'
data_dir = base_dir + '/data'
processed_dir = data_dir + '/processed'
plots_dir = base_dir + '/plots'

from pathlib import Path
for i in [data_dir, processed_dir, plots_dir]:
    Path(i).mkdir(parents=True, exist_ok=True)

#file_path = './data/expression_mRNA_17-Aug-2014.txt'
file_path = data_dir + '/expression_mRNA_17-Aug-2014.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='DataLoading')

    parser.add_argument('--save_X', action='store_true', default=False,
        dest='save_X', help='save the X matrix to a file (default: False)')
    parser.add_argument('--verbose', action='store_true', default=False,
        dest='verbose', help='print out information')

    return parser.parse_args()

# load into Anndata object
def load_data(args):
    counts = []
    gene_names = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i == 0:
                tissue = np.asarray(row, dtype=np.str)[2:]
            if i == 1:
                precise_clusters = np.asarray(row, dtype=np.str)[2:]
            if i == 8:
                clusters = np.asarray(row, dtype=np.str)[2:]
            if i >= 11:
                counts.append(row[1:])
                gene_names.append(row[0])
            
    # To do: remove these from here?
    cell_types, labels = np.unique(clusters, return_inverse=True)
    _, precise_labels = np.unique(precise_clusters, return_inverse=True)

    X = np.array(counts, dtype=np.int).T[1:]

    var = pd.DataFrame(index=gene_names)
    obs = pd.DataFrame()                  

    obs['clusters'] = clusters           
    obs['tissue'] = tissue
    obs['n_counts'] = np.sum(X, axis=1)
    obs['n_genes_expressed'] = np.sum(X>0, axis=1)
    var['n_cells'] = np.sum((X > 0), axis=0) # Check this works??

    if args.save_X:
        np.savetxt('X.txt', X, delimiter='\t', fmt='%i')

    adata = ad.AnnData(X, var=var, obs=obs, dtype='int32')

    return adata


def debug_data(adata, args):
    if args.verbose:
#         print (f'there are {len(gene_names)} genes, {len(clusters)} cells, {len(cell_types)} cell types to recover which are:\n{", ".join(cell_types)}')
        cell_types = np.unique(adata.obs["clusters"], return_inverse=False)
        print (f'there are {len(adata.var.index)} genes, {len(adata.obs)} cells, {len(cell_types)} cell types to recover which are:\n{", ".join(cell_types)}')

#         print (f'the raw data has shape ({len(data)}, {len(data[0])})')
#         print (f'X has shape {X.shape}, {adata.X.shape}')
        print (f'X has shape {adata.X.shape}')

        print (f'var has shape {adata.var.shape}')
        print (f'the last 30 elements of vars are:\n{adata.var.index[-30:]}')
        print (f'the last 30 elements of vars are:\n{adata.var_names[-30:].tolist()}')
        print (f'adata.var has type {type(adata.var)}')
        print (f'the third value of adata.var {adata.var_names[3]}')

        print (f'adata.obs has length {adata.n_obs}')
        print (f'adata.obs has last 10 values {adata.obs_names[:10]}')
        print (f'adata.obs has keys of {adata.obs.keys()}')
        print (f'adata.obs["clusters"] has values of {adata.obs["clusters"][:10]}')


def get_h5ad_filename(id):
    h5ad_filename = processed_dir + '/adata_' + id + '.h5ad'
    return h5ad_filename


def save_h5ad (adata, id):
    h5ad_filename = get_h5ad_filename(id)
    adata.write(h5ad_filename, compression='gzip')


def load_h5ad(id):
    h5ad_filename = get_h5ad_filename(id)
    adata = sc.read(h5ad_filename, first_column_names=True)
    return adata


def main():
    args = parse_args()
    print ('Running load.py')

    adata = load_data(args)
    debug_data(adata, args)
    save_h5ad(adata, 'raw')

# do not run the code when it is imported, unless adata_raw.h5ad does not exist
if path.exists(get_h5ad_filename('raw')):
    if __name__ == '__main__':
        main()
else:
    main()
