import torch
import torch.nn as nn
import numpy as np
import copy
import sparse
import time

from model_functions import iter_loadtxt, basket_GRU, basket_GRUX, linear_GRU, custom_BCE, custom_MSE, top_prod_acc, batch_generator, batch_generatorX, BCE

# Prior to running this file, make sure that prepare_GRU_inputs_instacart.py is run in the same session and that the base model basketGRU_instacart.pth is trained and saved

batch_size = 64
hidden_dim = 750
rc_dropout = False
fw_dropout = False
stepwise = False
w_dec = 0
num_epochs = 20
only_last_val = True
only_last_train = False

seq_dim = 3
hour_dim = 7*16-1
seq_hidden = 5
hour_hidden = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("basketGRU_instacart.pth")  # Load the base model without covariates 
base_model = basket_GRU(ntime,nprod,hidden_dim)
base_model.load_state_dict(checkpoint["model_state_dict"])

network = basket_GRUX(batch_size, ntime-1, nprod, hidden_dim, seq_dim, hour_dim, seq_hidden, hour_hidden) 
network.W_fx = base_model.W_fx
network.W_fh = base_model.W_fh
network.b_f  = base_model.b_f

network.W_mx = base_model.W_mx
network.W_mh = base_model.W_mh
network.b_m  = base_model.b_m

network.W_cx = base_model.W_cx
network.W_ch = base_model.W_ch
network.b_c  = base_model.b_c

network.W_ph = base_model.W_ph
network.b_p  = base_model.b_p

network = network.to(device)

checkpoint = copy.deepcopy(network)
bestep = 0

train_generator = batch_generatorX(train_input, train_hour_input, train_seq_input, train_output, train_last, batch_size, nusers, shuffle = True)
val_generator = batch_generatorX(val_input, val_hour_input, val_seq_input, val_output, val_last, batch_size, np.shape(val_users)[0], shuffle = True)
test_generator = batch_generatorX(test_input, test_hour_input, test_seq_input, test_output, test_last, batch_size, np.shape(test_users)[0], shuffle = True)

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
    (inputs, hour, seq, targets, seq_lengths) = next(train_generator)
    inputs = torch.from_numpy(inputs.todense()).float().to(device)
    hour = torch.from_numpy(hour.todense()).float().to(device)
    seq = torch.from_numpy(seq).float().to(device)
    targets = torch.from_numpy(targets.todense()).float().to(device)
    seq_lengths = torch.from_numpy(seq_lengths.astype('long')).long().to(device)
    
    pred = network(inputs,hour,seq,fw_dropout,rc_dropout,stepwise)
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
            (inputs, hour, seq, targets, seq_lengths) = next(val_generator)
            inputs = torch.from_numpy(inputs.todense()).float().to(device)
            hour = torch.from_numpy(hour.todense()).float().to(device)
            seq = torch.from_numpy(seq).float().to(device)
            targets = torch.from_numpy(targets.todense()).float().to(device)
            seq_lengths = torch.from_numpy(seq_lengths.astype('long')).long().to(device)

            val_pred = network(inputs,hour,seq)
            loss = val_criterion(val_pred,targets,seq_lengths)
            valloss += inputs.size()[0]*loss.item()
            valacc += inputs.size()[0]*top_prod_acc(val_pred,targets,seq_lengths)

        if (epoch_count == 0):
            checkpoint = copy.deepcopy(network)
            bestep = epoch_count
        elif (valacc/np.shape(val_users)[0]  > np.min(valacc[np.nonzero(val_loss)])):
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

print("bestep")
torch.save({'epoch': bestep,
            'model_state_dict': checkpoint.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vloss': val_loss,
            'tloss': train_loss,
            'vacc': val_acc,
            'tacc': train_acc
            }, "basketGRU_instacart_withcov.pth")
