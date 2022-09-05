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
all_coordinates = all_coordinates.astype('int64')

order_selection = iter_loadtxt("order_selection.csv", delimiter=',', skiprows=1)
order_selection = order_selection.astype('int64')
order_selection[:,1] = order_selection[:,1]-1
order_selection[:,3] = order_selection[:,3]-1

nusers = np.max(all_coordinates[:,0])+1
ntime = np.max(all_coordinates[:,1])+1
nprod = np.max(all_coordinates[:,2])+1

train_coordinates = all_coordinates[all_coordinates[:,3]==0,0:3]
val_coordinates = all_coordinates[all_coordinates[:,3]==1,0:3]
test_coordinates = all_coordinates[all_coordinates[:,3]==2,0:3]

train_orders = order_selection[order_selection[:,2]==0,:]
val_orders = order_selection[order_selection[:,2]==1,:]
test_orders = order_selection[order_selection[:,2]==2,:]

seq_variables = order_selection[:,np.array([1,3,6,7])].astype('float32')
seq_variables[:,1] = seq_variables[:,1]/(ntime-1)
seq_variables[:,2] = seq_variables[:,2]/30
seq_variables[:,3] = seq_variables[:,3]/365

seq_input = np.zeros((nusers,ntime-1,seq_dim))
for user in np.arange(nusers):
    seq = seq_variables[seq_variables[:,0]==user,1:]
    seq = seq[seq[:,0]>0,:] # covariates of first basket never serve as input
    seq_input[user,np.arange(seq.shape[0]),:] = seq       
    
train_sequences = sparse.COO(np.ndarray.transpose(train_coordinates),True,shape=(nusers,ntime-1,nprod))
train_last = np.sum(np.sum(train_sequences,2)>0,1).todense()-1
is_last = train_coordinates[:,1]==train_last[train_coordinates[:,0]]

train_input = sparse.COO(np.ndarray.transpose(train_coordinates[is_last==False]),True,shape=(nusers,ntime-2,nprod))
is_first = train_coordinates[:,1]==0
train_output = sparse.COO(np.ndarray.transpose(train_coordinates[is_first==False]),True,shape=(nusers,ntime-1,nprod))
train_output = train_output[:,1:(ntime-1),:]

train_hour = train_orders[:,np.array([1,3,9])] # select columns with user_id, order_number and part_of_week
train_hour = train_hour[train_hour[:,2]<111,:] # ignore dummy 111 for identification
train_hour_input = sparse.COO(np.ndarray.transpose(train_hour),True,shape=(nusers,ntime-1,hour_dim))
train_hour_input = train_hour_input[:,1:(ntime-1),:]

train_seq_input = np.copy(seq_input)
train_seq_input[np.arange(nusers),train_last,:] = 0
train_seq_input = train_seq_input[:,0:(ntime-2),:]

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

val_seq_input = seq_input[val_users,:,:]
test_seq_input = seq_input[test_users,:,:]

val_hour = val_orders[:,np.array([1,3,9])] # select columns with user_id, order_number and part_of_week
val_hour = val_hour[val_hour[:,2]<111,:] # ignore dummy 111 for identification
val_hour_input = sparse.COO(np.ndarray.transpose(val_hour),True,shape=(nusers,ntime,hour_dim))
val_hour_input = val_hour_input[val_users,1:ntime,:]

test_hour = test_orders[:,np.array([1,3,9])] # select columns with user_id, order_number and part_of_week
test_hour = test_hour[test_hour[:,2]<111,:] # ignore dummy 111 for identification
test_hour_input = sparse.COO(np.ndarray.transpose(test_hour),True,shape=(nusers,ntime,hour_dim))
test_hour_input = test_hour_input[test_users,1:ntime,:]



