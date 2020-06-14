#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:04:39 2020

@author: Elliott
"""

# Using Mouse Cortex Cells dataset
# 3005, gold-standard labels for seven distinct cell types
# Data available from https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt

import csv
import numpy as np


filename = './data/expression_mRNA_17-Aug-2014.txt'

rows = []
gene_names = []

with open(filename, "r") as csvfile:
    data_reader = csv.reader(csvfile, delimiter="\t")
    for i, row in enumerate(data_reader):
        if i == 1:
            precise_clusters = np.asarray(row, dtype=str)[2:]
        if i == 8:
            clusters = np.asarray(row, dtype=str)[2:]
        if i >= 11:
            rows.append(row[1:])
            gene_names.append(row[0])
            
cell_types, labels = np.unique(clusters, return_inverse=True)
_, precise_labels = np.unique(precise_clusters, return_inverse=True)