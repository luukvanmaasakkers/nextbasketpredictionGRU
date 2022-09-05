import torch
import torch.nn as nn
import numpy as np
import copy
import sparse
import time

from model_functions import iter_loadtxt, basket_GRU, basket_GRUX, linear_GRU, custom_BCE, custom_MSE, top_prod_acc, batch_generator, batch_generatorX, BCE

# Prior to running this file, prepare_GRU_inputs_instacart.py must be run in the same session

batch_size = 64
hidden_dim = 750
rc_dropout = False
fw_dropout = False
stepwise = False
w_dec = 0
num_epochs = 150
only_last_val = True
only_last_train = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = basket_GRU(ntime,nprod,hidden_dim) 
network = network.to(device)

checkpoint = copy.deepcopy(network)
bestep = 0

train_generator = batch_generator(train_input, train_output, train_last, batch_size, nusers, shuffle = True)
val_generator = batch_generator(val_input, val_output, val_last, batch_size, np.shape(val_users)[0], shuffle = True)
test_generator = batch_generator(test_input, test_output, test_last, batch_size, np.shape(test_users)[0], shuffle = True)

criterion = custom_BCE(only_last_train)
val_criterion = custom_BCE(only_last_val)
optimizer = torch.optim.Adam(network.parameters()) 

train_batches_per_epoch = 162 # validation set evaluation after approx. 10,000 baskets in total

val_batches_per_epoch = np.ceil(np.shape(val_users)[0]/batch_size)
num_iter = num_epochs*train_batches_per_epoch

iter_loss = 0
iter_acc = 0
iter_n = 0

epoch_count = 0
train_loss = np.zeros(num_epochs)
val_loss = np.zeros(num_epochs)
train_acc = np.zeros(num_epochs)
val_acc = np.zeros(num_epochs)

start_time = time.time()
print("Start network training")
for i in np.arange(num_iter):
    (inputs, targets, seq_lengths) = next(train_generator)
    inputs = torch.from_numpy(inputs.todense()).float().to(device)
    targets = torch.from_numpy(targets.todense()).float().to(device)
    seq_lengths = torch.from_numpy(seq_lengths.astype('long')).long().to(device)
    
    pred = network(inputs,fw_dropout,rc_dropout,stepwise)
    loss = criterion(pred,targets,seq_lengths)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = top_prod_acc(pred,targets,seq_lengths)
    iter_loss += inputs.size()[0]*loss.item()
    iter_acc += inputs.size()[0]*acc
    iter_n += inputs.size()[0]

    if ((i+1) % train_batches_per_epoch == 0):
        train_loss[epoch_count] = iter_loss/iter_n
        train_acc[epoch_count] = iter_acc/iter_n
        valloss = 0
        valacc = 0
        testloss = 0
        testacc = 0
        
        for j in np.arange(val_batches_per_epoch):
            (inputs, targets, seq_lengths) = next(val_generator)
            inputs = torch.from_numpy(inputs.todense()).float().to(device)
            targets = torch.from_numpy(targets.todense()).float().to(device)
            seq_lengths = torch.from_numpy(seq_lengths.astype('long')).long().to(device)

            val_pred = network(inputs)
            loss = val_criterion(val_pred,targets,seq_lengths)
            valloss += inputs.size()[0]*loss.item()
            valacc += inputs.size()[0]*top_prod_acc(val_pred,targets,seq_lengths)

        if (epoch_count == 0):
            checkpoint = copy.deepcopy(network)
            bestep = epoch_count
        elif (valacc/np.shape(val_users)[0]  > np.min(valacc[np.nonzero(valacc)])):
            checkpoint = copy.deepcopy(network)
            bestep = epoch_count
            
        val_loss[epoch_count] = valloss/np.shape(val_users)[0]    
        val_acc[epoch_count] = valacc/np.shape(val_users)[0]  
        iter_loss = 0
        iter_acc = 0
        iter_n = 0
        epoch_count += 1

        print("Epoch {}/{} ({:6} s): training loss = {:7}, validation loss = {:7}, training acc = {:7}, validation acc = {:7}".format(epoch_count,
                 num_epochs,np.round(time.time()-start_time,1),np.round(1000*train_loss[epoch_count-1],4),
                 np.round(1000*val_loss[epoch_count-1],4),np.round(train_acc[epoch_count-1],4),np.round(val_acc[epoch_count-1],4)))

torch.save({'epoch': epoch_count,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vloss': val_loss,
            'tloss': train_loss,
            'vacc': val_acc,
            'tacc': train_acc
            }, "basketGRU_instacart.pth") # saves the network, losses and accuracies
