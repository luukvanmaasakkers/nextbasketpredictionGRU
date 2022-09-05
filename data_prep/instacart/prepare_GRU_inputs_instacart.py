import numpy as np
import sparse
import torch
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd

from IPython.display import display, clear_output
from ipywidgets import Output

from functions import iter_loadtxt, basket_GRU, basket_GRUX, linear_GRU, custom_BCE, custom_MSE, top_prod_acc, batch_generator, batch_generatorX, BCE

all_coordinates = iter_loadtxt("all_3dcoordinates.csv", delimiter=',', skiprows=1) # all_3dcoordinates.csv is an output from cluster_products_instacart.R

departments = pd.read_csv("departments.csv", delimiter=',').department.tolist()  # departments.csv is part of the source files published on Kaggle
aisles = pd.read_csv("aisles.csv", delimiter=",").aisle.tolist() # aisles.csv is part of the source files published on Kaggle

all_coordinates = all_coordinates.astype('int64')

nusers = np.max(all_coordinates[:,0])+1
ntime = np.max(all_coordinates[:,1])+1
nprod = np.max(all_coordinates[:,2])+1

train_coordinates = all_coordinates[all_coordinates[:,3]==0,0:3]
val_coordinates = all_coordinates[all_coordinates[:,3]==1,0:3]
test_coordinates = all_coordinates[all_coordinates[:,3]==2,0:3]

train_sequences = sparse.COO(np.ndarray.transpose(train_coordinates),True,shape=(nusers,ntime-1,nprod))
train_last = np.sum(np.sum(train_sequences,2)>0,1).todense()-1
is_last = train_coordinates[:,1]==train_last[train_coordinates[:,0]]

train_input = sparse.COO(np.ndarray.transpose(train_coordinates[is_last==False]),True,shape=(nusers,ntime-2,nprod))
is_first = train_coordinates[:,1]==0
train_output = sparse.COO(np.ndarray.transpose(train_coordinates[is_first==False]),True,shape=(nusers,ntime-1,nprod))
train_output = train_output[:,1:(ntime-1),:]

complete_last = train_last+1
is_last = all_coordinates[:,1]==complete_last[all_coordinates[:,0]]
complete_input = sparse.COO(np.ndarray.transpose(all_coordinates[is_last==False,0:3]),True,shape=(nusers,ntime,nprod))
is_first = all_coordinates[:,1]==0
complete_output = sparse.COO(np.ndarray.transpose(all_coordinates[is_first==False,0:3]),True,shape=(nusers,ntime,nprod))
complete_output = complete_output[:,1:ntime,:]

val_users = np.unique(val_coordinates[:,0])
test_users = np.unique(test_coordinates[:,0])
val_input = complete_input[val_users,:,:]
test_input = complete_input[test_users,:,:]
val_output = complete_output[val_users,:,:]
test_output = complete_output[test_users,:,:]
val_last = complete_last[val_users]
test_last = complete_last[test_users]

