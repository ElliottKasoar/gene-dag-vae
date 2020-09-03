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
    
    return adata


def cluster_accuracy(adata):
    
    accuracy_1 = []
    accuracy_2 = []
    
    # categories = adata.obs['clusters'].cat.categories.tolist()
    categories_1 = adata.obs['clusters'].values.unique().tolist()
    
    for category in categories_1:
        l_clusters = adata[adata.obs['clusters'] == category].obs['leiden'].value_counts()
        accuracy_1.append(l_clusters.max() / l_clusters.sum())
        
    # categories = adata.obs['clusters'].cat.categories.tolist()
    categories_2 = adata.obs['leiden'].values.unique().tolist()
    
    for category in categories_2:
        l_clusters = adata[adata.obs['leiden'] == category].obs['clusters'].value_counts()
        accuracy_2.append(l_clusters.max() / l_clusters.sum())
    
    return accuracy_1, accuracy_2 


# Preprocessed plots 

# adata = load_h5ad('preprocessed')

# adata = cluster(adata, stage='preprocessed', color='clusters')
# adata = cluster(adata, stage='preprocessed', color='leiden', save_data=True)

# acc1, acc2 = cluster_accuracy(adata)
# print(acc1)
# print(acc2)


# Denoised plots
adata = load_h5ad('denoised')

adata = cluster(adata, stage='denoised', color='clusters')
adata = cluster(adata, stage='denoised', color='leiden', save_data=True)

acc3, acc4 = cluster_accuracy(adata)
print(acc3)
print(acc4)