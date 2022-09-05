import numpy as np
import torch
import torch.nn as nn
import sparse
import copy
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd

from model_functions import iter_loadtxt, basket_GRU, basket_GRUX, linear_GRU, custom_BCE, custom_MSE, top_prod_acc, batch_generator, batch_generatorX, BCE

# Before running this file, make sure that prepare_GRU_inputs_instacart.py is run in the same session 
# and the trained GRU model basketGRU_instacart_withcov.pth is obtained from the train_instacart_withcov.py file

def measures(pred,target):
    """
    Function that calculates the performance measures for a batch of predictions
    
    Arguments:
      pred      batch of predictions with size [batch size] x [assortment size]
      target    batch of target labels (1 = purchased, 0 = non-purchased) with same size as pred
    """
    pred_ranking = torch.sort(pred,dim=1,descending=True).indices
    target_items = target.nonzero()
    
    acc = 0
    all_ranks = 0
    
    for customer in np.arange(0,pred.size()[0]):
        
        customer_ranking = pred_ranking[customer,:]        
        target_prods = target_items[target_items[:,0]==customer,1]
        pred_prods = customer_ranking[0:target_prods.size()[0]]
        
        # Top rank accuracy for target
        diff = np.setdiff1d(target_prods,pred_prods)
        acc += 1-(np.shape(diff)[0]/target_prods.size()[0])
        
        # Average rank for target
        customer_rank = 0
        for product in target_prods:
            customer_rank += (customer_ranking==product).nonzero()
        customer_rank = customer_rank.double()/target_prods.size()[0]
        all_ranks += customer_rank             
    
    tra = acc/pred.size()[0]
    av_rank = all_ranks/pred.size()[0]
    return(tra,av_rank)

# Calculate benchmark predictions
# Generally most frequent
product_frequencies = np.sum(train_sequences[:,:,:],axis=np.array([0,1])).todense()
total_baskets = np.sum(np.sum(train_sequences[:,:,:],axis=2)>0)
product_proportions = product_frequencies/total_baskets
prop_pred = np.asarray(np.repeat(np.asmatrix(product_proportions),test_users.shape[0],axis=0))

# Personally most frequent
prior_test_sequences = train_sequences[test_users,:,:]
prior_nbaskets = train_last[test_users]+1
personal_frequencies = np.sum(prior_test_sequences,1).todense()
personal_frequencies_untied = personal_frequencies + np.repeat(np.asmatrix(product_proportions),test_users.shape[0],axis=0)
personal_prop_pred = personal_frequencies_untied/np.repeat(np.transpose(np.asmatrix(prior_nbaskets+1)),nprod,axis=1)
del personal_frequencies, personal_frequencies_untied

# Personal last basket
last_baskets = np.zeros(np.array([test_users.shape[0],nprod]))
for i in np.arange(test_users.shape[0]):
    last_baskets[i,:] = prior_test_sequences[i,prior_nbaskets[i]-1,:].todense()
last_baskets_untied = last_baskets + np.repeat(np.asmatrix(product_proportions),test_users.shape[0],axis=0)
last_baskets_pred = last_baskets_untied/np.repeat(2,nprod)
del last_baskets, last_baskets_untied

batch_size = 32
hidden_dim = 750
model = basket_GRUX(batch_size, ntime-1, nprod, hidden_dim, seq_dim, hour_dim, seq_hidden, hour_hidden)
test_generator = batch_generatorX(test_input, test_hour_input, test_seq_input, test_output, test_last, batch_size, np.shape(test_users)[0], shuffle = True)  

checkpoint = torch.load("drive/MyDrive/GRU revision/basketGRU_instacart.pth") # Load trained network
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

val_criterion = custom_BCE(only_last = True)
ntest_batches = np.ceil(test_users.shape[0]/batch_size)

testacc = 0
testrank = 0
testloss = 0

testacc_bench1 = 0
testrank_bench1 = 0
testbce1 = 0

testacc_bench2 = 0
testrank_bench2 = 0
testbce2 = 0

testacc_bench3 = 0
testrank_bench3 = 0
testbce3 = 0

out = Output()
for j in np.arange(ntest_batches):
  (selection, input, hour, seq, target, last) = next(test_generator)
  hour = torch.from_numpy(hour.todense()).float().to(device)
  seq = torch.from_numpy(seq).float().to(device)
  
  inputs = torch.from_numpy(input.todense()).float().to(device)
  targets = torch.from_numpy(target.todense()).float().to(device)
  seq_lengths = torch.from_numpy(last.astype('long')).long().to(device)
    
  test_pred = model(inputs, hour, seq)

  target_last = targets[torch.arange(targets.size()[0]),seq_lengths-1,:]
  pred_last = test_pred[torch.arange(targets.size()[0]),seq_lengths-1,:]
    
  loss = val_criterion(test_pred,targets,seq_lengths)
  testloss += inputs.size()[0]*loss.item()
      
  (tra, av_rank) = measures(pred_last.cpu(),target_last.cpu())

  testacc += tra*inputs.size()[0]
  testrank += av_rank*inputs.size()[0]
    
  #BENCHMARKS
  pred_bench1 = torch.from_numpy(prop_pred[selection,:])
  pred_bench2 = torch.from_numpy(personal_prop_pred[selection,:])
  pred_bench3 = torch.from_numpy(last_baskets_pred[selection,:]) 

  (tra, av_rank) = measures(pred_bench1.cpu(),target_last.cpu())

  testacc_bench1 += tra*inputs.size()[0]
  testrank_bench1 += av_rank*inputs.size()[0]
  testbce1 += BCE(prop_pred[selection,:],target_last.detach().cpu().numpy())*inputs.size()[0]

  (tra, av_rank) = measures(pred_bench2.cpu(),target_last.cpu())

  testacc_bench2 += tra*inputs.size()[0]
  testrank_bench2 += av_rank*inputs.size()[0]
  testbce2 += BCE(personal_prop_pred[selection,:],target_last.detach().cpu().numpy())*inputs.size()[0]

  (tra, av_rank) = measures(pred_bench3.cpu(),target_last.cpu())

  testacc_bench3 += tra*inputs.size()[0]
  testrank_bench3 += av_rank*inputs.size()[0]
  testbce3 += BCE(last_baskets_pred[selection,:],target_last.detach().cpu().numpy())*inputs.size()[0]
  
  with out:
    clear_output()
  out = Output()
  display(out)
  with out:
    print("Evaluating", np.round((j+1)/ntest_batches*100,1),"%")

print("MODEL PERFORMANCE")
print("Binary cross-entropy        ", np.round(testloss*1000/test_users.shape[0],2), "x10e-3")
print("Accuracy                    ", np.round(testacc/test_users.shape[0]*100,2),"%")
print("Average rank products       ", np.round(testrank.item()/test_users.shape[0],2))

print("POPULARITY BENCHMARK PERFORMANCE")
print("Binary cross-entropy        ", np.round(testbce1*1000/test_users.shape[0],2), "x10e-3")
print("Accuracy                    ", np.round(testacc_bench1/test_users.shape[0]*100,2),"%")
print("Average rank products       ", np.round(testrank_bench1.item()/test_users.shape[0],2))

print("INDIVIDUAL POPULARITY BENCHMARK PERFORMANCE")
print("Binary cross-entropy        ", np.round(testbce2*1000/test_users.shape[0],2), "x10e-3")
print("Accuracy                    ", np.round(testacc_bench2/test_users.shape[0]*100,2),"%")
print("Average rank products       ", np.round(testrank_bench2.item()/test_users.shape[0],2))

print("INDIVIDUAL LAST BASKET BENCHMARK PERFORMANCE")
print("Binary cross-entropy        ", np.round(testbce3*1000/test_users.shape[0],2), "x10e-3")
print("Accuracy                    ", np.round(testacc_bench3/test_users.shape[0]*100,2),"%")
print("Average rank products       ", np.round(testrank_bench3.item()/test_users.shape[0],2))
