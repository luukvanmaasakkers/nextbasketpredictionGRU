import numpy as np
import sparse
import torch
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd

from IPython.display import display, clear_output
from ipywidgets import Output

from model_functions import iter_loadtxt

dunnhumby = iter_loadtxt("transaction_data.csv", delimiter=',', skiprows=1).astype('int64') # csv file can be downloaded from Kaggle

products = dunnhumby[:,3]
prod_freq = np.unique(products,return_counts = True)
prods_unique = prod_freq[0]
prods_freq = prod_freq[1]
prod_select = prods_unique[prods_freq>=50]

dunnhumby_subset = dunnhumby[np.isin(dunnhumby[:,3],prod_select),:]
perc_lost = len(dunnhumby_subset)/len(dunnhumby)

all_orders = np.unique(dunnhumby_subset[:,0:2],axis=0)
household_freq = np.unique(all_orders[:,0],return_counts=True)
households_unique = household_freq[0]
households_freq = household_freq[1]
household_select = households_unique[households_freq>=3]

dunnhumby_subset = dunnhumby_subset[np.isin(dunnhumby_subset[:,0],household_select),:]

perc_lost = len(dunnhumby_subset)/len(dunnhumby)

households = np.unique(dunnhumby_subset[:,0])
new_households = np.zeros(len(dunnhumby_subset)).astype('int64')

for i in np.arange(len(households)):
  old_household = households[i]
  new_households[dunnhumby_subset[:,0]==old_household] = i

n_households = len(np.unique(new_households))
dunnhumby_subset[:,0] = new_households

products = np.unique(dunnhumby_subset[:,3])
new_products = np.zeros(len(dunnhumby_subset)).astype('int64')

for i in np.arange(len(products)):
  old_product = products[i]
  new_products[dunnhumby_subset[:,3]==old_product] = i

n_products = len(np.unique(new_products))
dunnhumby_subset[:,3] = new_products

order = np.lexsort((dunnhumby_subset[:,8], # trans_time
                   dunnhumby_subset[:,2], # day
                   dunnhumby_subset[:,0])) # household_id
dunnhumby_ord = dunnhumby_subset[order,:]

n_purch = len(dunnhumby_ord)
new_basket = np.full(n_purch, False)
new_household = np.full(n_purch, False)
new_basket[1:] = dunnhumby_ord[1:n_purch,1] != dunnhumby_ord[0:(n_purch-1),1]
new_household[1:] = dunnhumby_ord[1:n_purch,0] != dunnhumby_ord[0:(n_purch-1),0]

times = np.zeros(n_purch).astype('int64')
t = 0
for i in np.arange(n_purch):
  if new_household[i]:
    t = 0
  elif new_basket[i]:
    t += 1
  times[i] = t

dunnhumby_ord[:,1] = times

maxlen = 100
dunnhumby_trunc = dunnhumby_ord[dunnhumby_ord[:,1]<maxlen,:]
n_purch = len(dunnhumby_trunc)

n_bask_perh = np.zeros(n_households).astype('int64')
for h in np.arange(n_households):
  n_bask_perh[h] = np.max(dunnhumby_ord[dunnhumby_ord[:,0]==h,1])+1

perc_baskets_removed = np.mean(n_bask_perh>maxlen)

dunnhumby_coord = dunnhumby_trunc[:,np.array([0,1,3])]

first_of_basket = np.full(n_purch,True)
first_of_basket[1:] = dunnhumby_trunc[1:n_purch,1] != dunnhumby_trunc[0:(n_purch-1),1]
orders = dunnhumby_trunc[first_of_basket,:] 
dunnhumby_orders = orders[:,np.array([0,1,2,6,8,9])]# day, store_id, trans_time and week_no

nusers = np.max(dunnhumby_coord[:,0])+1
np.random.seed(123)
test_users = np.sort(np.random.choice(np.arange(nusers),int(np.floor(nusers/2)),replace=False))

orders_set = np.zeros(len(dunnhumby_orders)).astype('int64')
last_of_user = np.full(len(dunnhumby_orders),True)
last_of_user[0:(len(last_of_user)-1)] = dunnhumby_orders[0:(len(last_of_user)-1),0] != dunnhumby_orders[1:len(last_of_user),0]
is_user_in_test = np.isin(dunnhumby_orders[:,0],test_users)

orders_set[last_of_user & ~is_user_in_test] = 1 # validation orders
orders_set[last_of_user & is_user_in_test] = 2 # test orders
orders_set = np.expand_dims(orders_set,1)
order_ids = np.expand_dims(np.arange(len(orders_set)),1)
dunnhumby_orders = np.concatenate((order_ids,dunnhumby_orders,orders_set),1)

order_id = np.zeros(len(dunnhumby_coord)).astype('int64')
count = 0
new_order = np.full(len(dunnhumby_coord),False)
new_order[1:] = dunnhumby_coord[1:,1] != dunnhumby_coord[0:(len(dunnhumby_coord)-1),1]
for i in np.arange(len(dunnhumby_coord)):
  if new_order[i]:
    count += 1
  order_id[i] = count

purchase_set = np.expand_dims(dunnhumby_orders[order_id,7],1)
order_id = np.expand_dims(order_id,1)
dunnhumby_coord = np.concatenate((dunnhumby_coord,order_id,purchase_set),1)

trunc = False # Can be set to True to obtain predictions based on only the 50 most recent baskets (similar to TARS)
if trunc:
  maxlen = 51
  nusers = np.max(dunnhumby_coord[:,0])+1
  ntime = np.max(dunnhumby_coord[:,1])+1
  nprod = np.max(dunnhumby_coord[:,2])+1
  
  train_coordinates = dunnhumby_coord[dunnhumby_coord[:,4]==0,0:3]
  train_sequences = sparse.COO(np.ndarray.transpose(train_coordinates),True,shape=(nusers,ntime-1,nprod))
  complete_last = np.sum(np.sum(train_sequences,2)>0,1).todense()

  trunc_coord = np.zeros((0,5)).astype('int64')
  for i in np.arange(nusers):
    selection = dunnhumby_coord[:,0] == i
    old_coords = dunnhumby_coord[selection,:]
    n_baskets = complete_last[i]+1
    if n_baskets > maxlen:
      select_mostrecent = old_coords[:,1] >= n_baskets-maxlen
      old_coords = old_coords[select_mostrecent]
      old_coords[:,1] = old_coords[:,1] - (n_baskets-maxlen)
    trunc_coord = np.concatenate((trunc_coord,old_coords),0)
    if i % 100 == 0:
      print(i)
  dunnhumby_coord = trunc_coord

train_orders = dunnhumby_orders[dunnhumby_orders[:,7]==0,:]
test_orders = dunnhumby_orders[dunnhumby_orders[:,7]==1,:]
val_orders = dunnhumby_orders[dunnhumby_orders[:,7]==2,:]  

train_coordinates = dunnhumby_coord[dunnhumby_coord[:,4]==0,0:3]
test_coordinates = dunnhumby_coord[dunnhumby_coord[:,4]==1,0:3]
val_coordinates = dunnhumby_coord[dunnhumby_coord[:,4]==2,0:3]

nusers = np.max(dunnhumby_coord[:,0])+1
ntime = np.max(dunnhumby_coord[:,1])+1
nprod = np.max(dunnhumby_coord[:,2])+1

train_sequences = sparse.COO(np.ndarray.transpose(train_coordinates),True,shape=(nusers,ntime-1,nprod))
train_last = np.sum(np.sum(train_sequences,2)>0,1).todense()-1
is_last = train_coordinates[:,1]==train_last[train_coordinates[:,0]]

train_input = sparse.COO(np.ndarray.transpose(train_coordinates[is_last==False]),True,shape=(nusers,ntime-2,nprod))
is_first = train_coordinates[:,1]==0
train_output = sparse.COO(np.ndarray.transpose(train_coordinates[is_first==False]),True,shape=(nusers,ntime-1,nprod))
train_output = train_output[:,1:(ntime-1),:]

complete_last = train_last+1
is_last = dunnhumby_coord[:,1]==complete_last[dunnhumby_coord[:,0]]
complete_input = sparse.COO(np.ndarray.transpose(dunnhumby_coord[is_last==False,0:3]),True,shape=(nusers,ntime-1,nprod))
is_first = dunnhumby_coord[:,1]==0
complete_output = sparse.COO(np.ndarray.transpose(dunnhumby_coord[is_first==False,0:3]),True,shape=(nusers,ntime,nprod))
complete_output = complete_output[:,1:ntime,:]

val_users = np.unique(val_coordinates[:,0])
test_users = np.unique(test_coordinates[:,0])
val_input = complete_input[val_users,:,:]
test_input = complete_input[test_users,:,:]
val_output = complete_output[val_users,:,:]
test_output = complete_output[test_users,:,:]
val_last = complete_last[val_users]
test_last = complete_last[test_users]

hour_of_day = np.floor(dunnhumby_orders[:,5]/100).astype('int64')

night = np.array([23,0,1,2,3,4,5])
morning = np.array([6,7])
daytime = np.arange(8,21)
evening = np.array([21,22])
time_of_day = hour_of_day
time_of_day[np.isin(hour_of_day,night)] = 0
time_of_day[np.isin(hour_of_day,morning)] = 1
time_of_day[np.isin(hour_of_day,daytime)] = hour_of_day[np.isin(hour_of_day,daytime)] - 6
time_of_day[np.isin(hour_of_day,evening)] = 15 

purchase_day = dunnhumby_orders[:,3]
day_of_week = purchase_day % 7 

time_of_week = day_of_week*16 + time_of_day
time_of_week = np.expand_dims(time_of_week,1)

first_of_user = np.full(len(dunnhumby_orders),True)
first_of_user[1:] = dunnhumby_orders[1:,1] != dunnhumby_orders[0:(len(first_of_user)-1),1]

days_since_prior = np.zeros(len(dunnhumby_orders)).astype('int64')
days_since_prior[1:] = dunnhumby_orders[1:,3] - dunnhumby_orders[0:(len(days_since_prior)-1),3]
days_since_prior[first_of_user] = 0
capped_days_since_prior = np.copy(days_since_prior)
capped_days_since_prior[days_since_prior>50] = 50 # cap at 50

cumdays_since_first = np.zeros(len(days_since_prior)).astype('int64')
total = 0
for i in np.arange(1,len(days_since_prior)):
  if dunnhumby_orders[i,1] != dunnhumby_orders[i-1,1]:
    total = 0 
  else:
    total += days_since_prior[i]
    cumdays_since_first[i] = total

day_of_year = dunnhumby_orders[:,3] % 365
month_of_year = np.floor(day_of_year / (365/12)).astype('int64')

rel_timestamp = dunnhumby_orders[:,2:3]/np.max(dunnhumby_orders[:,2:3]) # divide by maximum timestamp (99) to normalize
capped_days_since_prior = np.expand_dims(capped_days_since_prior,1)/np.max(capped_days_since_prior)
cumdays_since_first = np.expand_dims(cumdays_since_first,1)/np.max(cumdays_since_first)

month_of_year_pd = pd.DataFrame(month_of_year)
month_dummies = pd.get_dummies(month_of_year_pd.astype(str)).to_numpy()
month_dummies = month_dummies[:,[0,1,4,5,6,7,8,9,10,11,2]] # rearrange, exlude last dummy column for identification

seq_variables = np.concatenate((rel_timestamp,capped_days_since_prior,cumdays_since_first,month_dummies),1)
seq_dim = seq_variables.shape[1]

seq_input = np.zeros((nusers,ntime-1,seq_dim))
for user in np.arange(nusers):
    seq = seq_variables[dunnhumby_orders[:,0]==user,:]
    seq = seq[seq[:,0]>0,:] # covariates of first basket never serve as input
    seq_input[user,np.arange(seq.shape[0]),:] = seq     

hour_dim = 111
hour_input = np.concatenate((dunnhumby_orders[:,1:3],time_of_week),1)

train_hour = hour_input[dunnhumby_orders[:,7]==0,:]
train_hour = train_hour[train_hour[:,2]<hour_dim,:] # ignore dummy 111 for identification
train_hour_input = sparse.COO(np.ndarray.transpose(train_hour),True,shape=(nusers,ntime-1,hour_dim))
train_hour_input = train_hour_input[:,1:(ntime-1),:]

train_seq_input = np.copy(seq_input)
train_seq_input[np.arange(nusers),train_last,:] = 0
train_seq_input = train_seq_input[:,0:(ntime-2),:]

val_seq_input = seq_input[val_users,:,:]
test_seq_input = seq_input[test_users,:,:]

val_hour = hour_input[dunnhumby_orders[:,7]==2,:]
val_hour = val_hour[val_hour[:,2]<hour_dim,:] # ignore dummy 111 for identification
val_hour_input = sparse.COO(np.ndarray.transpose(val_hour),True,shape=(nusers,ntime,hour_dim))
val_hour_input = val_hour_input[val_users,1:ntime,:]

test_hour = hour_input[dunnhumby_orders[:,7]==1,:]
test_hour = test_hour[test_hour[:,2]<hour_dim,:] # ignore dummy 111 for identification
test_hour_input = sparse.COO(np.ndarray.transpose(test_hour),True,shape=(nusers,ntime,hour_dim))
test_hour_input = test_hour_input[test_users,1:ntime,:]
