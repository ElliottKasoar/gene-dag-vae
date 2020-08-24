#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
from load import save_h5ad, load_h5ad
#from loss import NB_loglikelihood
from temp import save_figure, plotTSNE

def cluster(adata, stage='preprocessed', color='clusters', random_state=10,
            n_pcs=50, save_data=False):
      
    plot_id = f'{stage}-{color}'
    
    sc.tl.tsne(adata, use_rep='X', random_state=random_state, n_pcs=n_pcs)

    # TSNE plot of data, points labelled with ground truth cluster    
    if color == 'clusters':

        plotTSNE(adata, color=[color], plot_id=plot_id)

    else:
        
        sc.pp.neighbors(adata, use_rep='X', random_state=random_state,
                        n_pcs=n_pcs)
        
        sc.tl.leiden(adata)
        groups = adata.obs[color]
        plotTSNE(adata, color=[color], plot_id=plot_id)
    
    if save_data:
        save_h5ad(adata, 'clustered')


def cluster_accuracy(adata):
    
    accuracy = []
    
    # categories = adata.obs['clusters'].cat.categories.tolist()
    categories = adata.obs['clusters'].values.unique().tolist()
    
    for category in categories:
        l_clusters = adata[adata.obs['clusters'] == category].obs['leiden'].value_counts()
        accuracy.append(l_clusters.max() / l_clusters.sum())
        
    return accuracy


# Preprocessed plots 
cluster(load_h5ad('preprocessed'), stage='preprocessed', color='clusters')
cluster(load_h5ad('preprocessed'), stage='preprocessed', color='leiden',
        save_data=True)

acc = cluster_accuracy(load_h5ad('clustered'))
print(acc)

# Denoised plots
cluster(load_h5ad('denoised'), stage='denoised', color='clusters')
cluster(load_h5ad('denoised'), stage='denoised', color='leiden',
        save_data=True)

acc = cluster_accuracy(load_h5ad('clustered'))
print(acc)