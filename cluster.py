#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc
from load import save_h5ad, load_h5ad
#from loss import NB_loglikelihood
from temp import save_figure, plotTSNE
import matplotlib.pyplot as plt
import numpy as np


def calculate_leiden_clusters(adata, stage='preprocessed', random_state=10,
            n_pcs=50, save_data=False):

    sc.pp.neighbors(adata, use_rep='X', random_state=random_state,
                        n_pcs=n_pcs)
        
    sc.tl.leiden(adata)

    if save_data:
        save_h5ad(adata, stage+'clustered')
    
    return adata

# calculate the accuracy of the calculated leiden clusters
def cluster_accuracy(adata, stage='preprocessed', random_state=10,
                     n_pcs=50, save_data=False):
    
    assert 'leiden' in adata.obs.keys().to_list(), 'must run calculate_leiden_clusters before clustering accuracy can be measured'
    
    accuracy_1 = []
    accuracy_2 = []
    
    # list of ground truth labels
    # categories = adata.obs['clusters'].cat.categories.tolist()
    categories_1 = adata.obs['clusters'].values.unique().tolist()
    
    # accuracy 1 is the maximum fraction of cells with the same ground truth that have the same leiden label
    for category in categories_1:
        l_clusters = adata[adata.obs['clusters'] == category].obs['leiden'].value_counts()
        accuracy_1.append(l_clusters.max() / l_clusters.sum())
    
    # list of calculated leiden labels
    # categories = adata.obs['clusters'].cat.categories.tolist()
    categories_2 = adata.obs['leiden'].values.unique().tolist()
    
    # accuracy 2 is the maximum fraction of cells with the same leiden label that have the same ground truth
    for category in categories_2:
        l_clusters = adata[adata.obs['leiden'] == category].obs['clusters'].value_counts()
        accuracy_2.append(l_clusters.max() / l_clusters.sum())
    
    return accuracy_1, accuracy_2


def plotGenesGroups(adata, stage='preprocessed', color = 'clusters', height=8, show=False):

    assert 'leiden' in adata.obs.keys().to_list(), 'must run calculate_leiden_clusters before clustering accuracy can be measured'
    
    print ('PLOTTING: genes groups')
    
    cols = 1
    width = height*cols
    
    # Finding marker genes
    sc.tl.rank_genes_groups(adata, groupby=color, method='wilcoxon', corr_method='bonferroni')
    save_h5ad(adata, stage+'after_marker_genes')
    
    # need to stop sc.pl from showing plot!
    fig, ax = plt.subplots(1,cols,figsize=(width,height))
    
    sc.pl.rank_genes_groups(adata,
                            n_genes=25,
                            sharey=False,
                            cols = 4,
                            ax=ax,
                            show=None)
    
    if show==True: plt.show(fig)
    save_figure('marker_genes', fig=fig)  
    plt.close(fig)
    
    print ('PLOTTING: genes groups heatmap')
    
    fig, ax = plt.subplots(1,cols,figsize=(width, height))
    sc.pl.rank_genes_groups_matrixplot(adata,
                                        #n_genes=25,
                                        groupby=color,
                                        use_raw=False,
                                        swap_axes=False if stage=='encoded' else True,
                                        figsize=(30,50),
                                        dendrogram=False)
    
    if show==True: plt.show(fig)
    save_figure('heatmap', fig=fig)  
    plt.close(fig)
    
    return None


def analyse_clusters(stage='preprocessed', color='cluster'):
    
    adata = load_h5ad(stage)
    
    adata = calculate_leiden_clusters(adata, stage=stage, random_state=10,
                               n_pcs=50, save_data=False)
    
    # compare number of ground truth clusters and number of leiden clusters
    print(len(adata.obs['clusters'].values.unique()),
          len(adata.obs['leiden'].values.unique()))
    
    acc1, acc2 = cluster_accuracy(adata, stage=stage, random_state=10,
                     n_pcs=50, save_data=False)
    print(acc1, '\n', acc2)
    print(np.mean(acc1), np.std(acc1))
    print(np.mean(acc2), np.std(acc2))
    
    if color == 'encoded':
        sc.tl.tsne(adata, use_rep='X', random_state=10, n_pcs=50)
    else:
        sc.tl.pca(adata, return_info=False, use_highly_variable=False)
        sc.tl.tsne(adata, use_rep='X_pca', random_state=10, n_pcs=50)
        
    plot_id = f'{stage}-{color}'
    plotTSNE(adata, color=[color], plot_id=plot_id, show=True)
    
    plotGenesGroups(adata, stage, show=True)
    
    
def main():
    
    stage = 'encoded'
    #stage = 'denoised'
    #stage = 'preprocessed'
    color='clusters'
    #color='leiden'
    
    analyse_clusters(stage=stage, color=color)
    
   
main()