# create feature matrix, adjacency matrix to be passed into graph convolutional layer

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import linregress
from load import load_h5ad, save_h5ad


# create directory 'models' if it doesn't exist
base_dir = '.'
data_dir = base_dir + '/data'
plots_dir = base_dir + '/plots'
models_dir = plots_dir + '/models'
processed_dir = data_dir + '/processed'

from pathlib import Path
for i in [plots_dir, models_dir]:
    Path(i).mkdir(parents=True, exist_ok=True)

adata = load_h5ad('denoised')    # need to add code to ensure this exists

# feature matrix: nodes are cells, features are the gene expression = adata.X
# binary adjacency matrix: nodes are cells, links are between cells that are correlated >> undirected

'''
adj = np.zeros((adata.n_obs, adata.n_obs))

# pearson correlation coeff, p value <= 0.5 statistically significant
# self loops not added

for i in range(adata.n_obs-1):
    dic = {}
    for j in range(i+1, adata.n_obs):
        __,__,r,p_value,__ = linregress(adata.X[i,:], adata.X[j,:])
        if p_value <= 0.5 and r >= 0:
            dic.update( {j : r} )
    print (len(dic.keys()))
            
    if len(dic.keys()) == 0:
        print ('No connections')
        
    else:
        max_value = max(dic.values())
        max_keys = [k for k, v in dic.items() if v == max_value]
        print (f'{i} has max r values of {max_value} at positions: {max_keys}')
        for k in max_keys:
            adj[i, k] = 1
    
# how many links from each cell?
print('how many entries: ', np.sum(adj[adj>0]))
print ('adj:', adj)
'''

# this currently assumes that the cells are already ordered (i.e. cell 0 is the first cell)
def calculate_edges(adata):
    edges = {}
    for i in range(adata.n_obs-1):
        dic = {}
        for j in range(i+1, adata.n_obs):
            __,__,r,p_value,__ = linregress(adata.X[i,:], adata.X[j,:])
            # expect similar cells to have a positive correlation
            if p_value <= 0.5 and r >= 0:
                dic.update( {j : r} )
            
        if len(dic.keys()) == 0:
            print ('No connections')
        
        else:
            max_value = max(dic.values())
            max_keys = [k for k, v in dic.items() if v == max_value]
            #print (f'{i} has max r values of {max_value} at positions: {max_keys}')
            for k in max_keys:
                edges.update( {i : k} )
    return edges

edges = calculate_edges(adata)
print (edges)
           
#adj = sp.coo_matrix((np.ones(len(edges)),([i for i in range(adata.n_obs-1)], [max_key for k])), shape=(adata.n_obs, adata.n_obs))

A = sp.coo_matrix((np.ones(len(edges)), (list(edges.keys()), 
                list(edges.values()))), shape=(adata.n_obs, adata.n_obs))            

###############################################################################

#adj = adj + np.ones((adata.n_obs, adata.n_obs)).multiply(adj.T)
#A += A.T

# print ('adj:', adj.toarray())
# print ('A:', A.toarray())

# print ('nonzero positions in adj:', np.nonzero(adj.toarray()))
# print ('nonzero positions in A:', np.nonzero(A.toarray()))

#%%
# if symmetric, D^-1/2 A D^-1/2, else D^-1 A
def normalise_adj(adj, symmetric=True):
    if symmetric:
        #sp.diags takes lists entries to put on the diagonal + list of offsets (k=0, k<0, k>0 tells you the position of each list of diags)
        # D = sp.diags(np.array(adj.sum(axis=1)), 0)       
        d = sp.diags(np.power( np.array(adj.sum(axis=1)), -0.5 ).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()      # adj, d both symmetric
    else:
        d = sp.diags(np.power (np.array(adj.sum(axis=1)), -1 ).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalise_adj(adj, symmetric)
    return adj

A_hat = preprocess_adj(A)

#%%
#graph = [X, A]


