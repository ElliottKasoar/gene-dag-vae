# temporary file to practice using adata, plots, preprocessing
# preprocessing to remove technical errors

from load import save_h5ad, load_h5ad
#from load import adata

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def display (obj, adata):
    [print (f"{item} has type:\t{type(obj[item])}") for item in obj]
    print (f'adata: {adata}')                # could replace everything with this?
    print (f'{adata.shape}')
#    print (f'{adata[:3, 'Tspan12'].X}')
    print (f'shape of the data matrix:\t{adata.X.shape}')    # or adata.shape
#        print (f'adata.var:\t{adata.var}')
    print (f'the info axis of the vars:\t{adata.var.keys()}')
    print (f'the info axis of the obs:\t{adata.obs.keys()}')
    print (f'number of vars (genes):\t{adata.n_vars}')
    print (f'numbers of obs (cells):\t{adata.n_obs}')
    print (f'the genes:\t{adata.var.index[:10].tolist()}')    # or adata.var_names
    print (f'the cells:\t{adata.obs.index[:10].tolist()}')
    print (f'the cluster each cell is taken from:\t{adata.obs["clusters"][:10].tolist()}')
    print (f'the tissue each cell is taken from:\t{adata.obs["tissue"][:10].tolist()}')
    print (f'the total number of counts in each cell:\t{adata.obs["n_counts"][:10].tolist()}')
    print (f'the total number genes expressed in each cell:\t{adata.obs["n_genes_expressed"][:10].tolist()}')
    print (f'adata.layers:\t{adata.layers}')

    
def examine_adata (adata):      # print out aggregates
    print (f'the number of zero counts:\t{np.sum(adata.X == 0)}')
    print (f'the number of counts:\t{adata.X.size}, {adata.n_obs*adata.n_vars}')
    print (f'the fraction of zero counts:\t{np.sum(adata.X == 0) / adata.X.size}')

    print (f'{adata.to_df().iloc[:5, :5]}')
    print (f'the number of Tspan12s are:\t{np.sum(adata.var.index == "Tspan12")}')
        
    print (f'mean expression of each cell:\n{adata.to_df().mean(axis=1).head()}')
    print (f'std of each cell:\n{adata.to_df().std(axis=1).head()}')
    print (f'median of each cell:\n{adata.to_df().median(axis=1).head()}')
    
    print (f'mean expression of each gene:\n{adata.to_df().mean(axis=0).head()}')
    print (f'std of each gene:\n{adata.to_df().std(axis=0).head()}')
    print (f'median of each gene:\n{adata.to_df().median(axis=0).head()}')
        
    print (f'the number of cells whose median expression is not 0:\t {np.sum(adata.to_df().median(axis=1) != 0)}')
    print (f'the number of counts in each cell:\n{np.sum(adata.X, axis=1)}')
    print (f'the median number of counts across the cells is:\t{np.median(adata.n_obs)}')
        
    #print (f'describe the genes:\n{adata.to_df().describe().iloc[:5]}')

# code to create the directory 'plots' if it doesnt exist
#base_dir = '.'
base_dir = '/home/chiaretta/performance/programming/github/GeneVAE'
data_dir = base_dir + '/data'
processed_dir = data_dir + '/processed'
plots_dir = base_dir + '/plots'

from pathlib import Path
for i in [data_dir, processed_dir, plots_dir]:
    Path(i).mkdir(parents=True, exist_ok=True)


def save_figure (id, fig=None, plt=None):
    filename = plots_dir + '/' + id + '.png'
    if fig is not None:
        fig.savefig(filename)
    elif plt is not None:
        plt.savefig(filename)

    
def plotHighestExprGenes(adata, n_top=20, show=False):

    height = (n_top * 0.2) + 1.5
    width = 5
        
    fig, ax = plt.subplots(figsize=(width,height))
    ax.set_title(f'Top {n_top} genes with the highest mean fractional count across all cells')
        
    sns.set(font_scale=1.5)
    sns.set_style("white")

    print ('PLOTTING: highest_expr_genes')

    sc.pl.highest_expr_genes(adata, n_top=n_top, ax=ax, show=False)

    sns.despine(offset=10, trim=False)
    plt.tight_layout()

    save_figure('highest_expr_genes', fig=fig)
    if show==True: plt.show(fig)
    plt.close(fig)
    

def plotViolin(adata, keysDict={}, height=8, show=False):

    cols = len(keysDict)
    fig, ax = plt.subplots(1,cols,figsize=(height*cols,height))

    print ('PLOTTING: violin_plot')

    for idx, item in enumerate(keysDict):
        axis = ax if cols==1 else ax[idx]

        axis.set_title(f'total {item} per cell')
        sc.pl.violin(adata, keysDict[item], ax=axis, show=False)

    if show==True: plt.show(fig)
    save_figure('violin_plot', fig=fig)    
    plt.close(fig)


def plotHistogram(adata, keysDict={}, bins=50, height=6, show=False):

    cols  = len(keysDict)
    fig, ax = plt.subplots(1,cols,figsize=(height*cols,height))

    sns.set(font_scale=1.5)
    sns.set_style("white")

    print ('PLOTTING: histogram')

    for idx, item in enumerate(keysDict):
        axis = ax if cols==1 else ax[idx]

        #axis.set_title(f'total {item} per cell')
        axis.set_title(f'{item}')

        # To do: change this so the format is the same as plotViolin
        #sns.distplot(adata.obs[keysDict[item][0]],
        sns.distplot(keysDict[item][0],
                 bins=bins,
                 color='black',
                 hist=True,
                 kde=False,
                 ax=axis)
        axis.axvline(x=keysDict[item][1], color='r', linestyle='--')

    if show==True: plt.show(fig)
    save_figure('histogram', fig=fig)  
    plt.close(fig)

    
def plotScatter(adata, pointSize=150, height=8, palette=sns.color_palette("deep"), show=False):

    print ('PLOTTING: scatter')

    cols = 1
    width  = height*cols

    fig, ax = plt.subplots(1,cols,figsize=(width,height))
    sns.set(font_scale=1.5)
    sns.set_style("white")

    sc.pl.scatter(adata, x='n_counts', y='n_genes_expressed',
                 color="tissue", palette=palette,
                 alpha=0.3, ax=ax, size=pointSize, show=False)


    sns.despine(offset=10, trim=False)
    plt.tight_layout()

    save_figure ('scatter', fig=fig)
    if show==True: plt.show(fig)
    plt.close(fig)


# this gives nan values since every cell has at least one zero count
# static calculation of size_factors using the 'median ratio method': Eqn 5 in Anders and Huber (2010)
def static_sf(adata):
    print ('calculating static size factors')

    products = np.prod(adata.X, axis=0)
    print (f'products: {products[:10]}')
    print (f'length of products: {len(products)}')

    ratios = adata.X/products            # broadcasting    
    print (f'ratios shape: {ratios.shape}')

    size_factors = np.median(ratios, axis=1)    # cell-specific size factors
    return size_factors


# revised method for calculating size factors
def calculate_sf(adata):
    print ('calculating size factors')

    size_factors = adata.obs['n_counts'] / np.mean(adata.X, axis=1)
    return size_factors

# [listVariables] is the list of variables you wish to plot
def plotPCA(adata, listVariables=[], pointSize=150, width=8, height=8, cols=2, palette=sns.color_palette("deep"), plot_id = 'PCA', show=False):

    print ('PLOTTING: pca')

    if len(listVariables) > 1:
        rows = int(len(listVariables)/cols)

        if rows*cols < len(listVariables):
            rows += 1
        
    else:
        rows = 1
        cols = 1

    fig, ax = plt.subplots(rows,cols,figsize=(width*cols,height*rows))
    sns.set(font_scale=1.5)
    sns.set_style("white")

    idx = 0
    for r in range(0, rows):
        for c in range(0, cols):

            if idx > len(listVariables):
                break
    
            if cols==1 and rows==1:
                axis=ax
            elif cols==1 or rows==1:
                axis=ax[idx]
            else:
                axis=ax[r,c] 
        
            var = listVariables[idx]
        
            if var == '':
                sc.pl.pca(adata,
                size=pointSize,
                palette=palette,
                ax=axis,
                show=False)
            else:
                sc.pl.pca(adata,
                color=var,
                size=pointSize,
                palette=palette,
                ax=axis,
                show=False)
            
            idx += 1
        
    sns.despine(offset=10, trim=False)
    plt.tight_layout()

    save_figure (plot_id, fig=fig)
    if show==True: plt.show(fig)
    plt.close(fig)    
 
def plotTSNE(adata, color, pointSize=150, height=8, palette=sns.color_palette("deep"), plot_id = 'TSNE', show=False):

    cols = len(color)
    width  = height*cols

    fig, ax = plt.subplots(1, cols, figsize=(width,height))

    print ('PLOTTING: tsne')

    # may change this so that color is a Dictionary
    for i in range(len(color)):
        # plt.subplots by default reduces array of Axes to a single axes if row and col=1
        if cols==1:
            sc.pl.tsne(adata,
            color=color,
            size=pointSize,
            palette=palette,
            ax=ax,
            show=False)
        
        else:
            sc.pl.tsne(adata,
            color=color[i],
            size=pointSize,
            palette=palette,
            ax=ax[i],
            show=False)
            
    save_figure (plot_id, fig=fig)
    if show==True: plt.show(fig)
    plt.close(fig)

def preprocessData(adata):
    adata = adata[adata.obs.n_genes_expressed > 200, :]
    adata = adata[:, adata.var['n_cells'] > 2]
    adata = adata[adata.obs.n_genes_expressed < 2500, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)    

    return adata

def main():
    adata = load_h5ad('raw')
    adata = preprocessData(adata)

    #adata.obs['static_sf'] = static_sf(adata)
    adata.obs['sf'] = calculate_sf(adata)

    save_h5ad(adata, 'preprocessed')

    display ({'adata.X':adata.X,
              'adata.var':adata.var,
              'adata.obs':adata.obs,
              'adata.uns':adata.uns},
              adata)

    #examine_adata (adata)


if __name__ == '__main__':
    main()

