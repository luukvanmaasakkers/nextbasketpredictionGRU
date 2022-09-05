import numpy as np
import torch
import torch.nn as nn

class basket_GRU(nn.Module):
  """
  Class for the basket GRU without covariates, as described by Van Maasakkers, Donkers en Fok (2022)
  """

  def __init__(self,seq_length,input_dim,num_hidden):
    """
    Initializes a basket_GRU instance
    
    Arguments:
      seq_length    number of time steps in each (padded) sequence, scalar
      input_dim     dimensionality of the GRU input (equals the assortment size here)
      num_hidden    dimensionality of the hidden GRU state, scalar hyperparameter
    """
    super(basket_GRU,self).__init__()
    self.seq_length = seq_length
    self.input_dim = input_dim
    self.num_hidden = num_hidden

    self.h_init = torch.zeros(num_hidden)

    # Forget gate parameters
    self.W_fx = nn.Parameter(torch.Tensor(input_dim,num_hidden))
    nn.init.xavier_uniform_(self.W_fx.data)
    self.W_fh = nn.Parameter(torch.Tensor(num_hidden,num_hidden))
    nn.init.xavier_uniform_(self.W_fh.data)
    self.b_f = nn.Parameter(torch.zeros(num_hidden))

    # Input modulator gate parameters
    self.W_mx = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_mx.data)
    self.W_mh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_mh.data)
    self.b_m = nn.Parameter(torch.zeros(num_hidden))
        
    # Input candidate gate parameters
    self.W_cx = nn.Parameter(torch.Tensor(input_dim, num_hidden))
    nn.init.xavier_uniform_(self.W_cx.data)
    self.W_ch = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
    nn.init.xavier_uniform_(self.W_ch.data)
    self.b_c = nn.Parameter(torch.zeros(num_hidden))
        
    # Output layer parameters
    self.W_ph = nn.Parameter(torch.Tensor(num_hidden, input_dim))
    nn.init.xavier_uniform_(self.W_ph.data)
    self.b_p = nn.Parameter(torch.zeros(input_dim))

  def forward(self, x, fw_dropout = False, rc_dropout = False, stepwise = False, track_hiddens = False):
    """
    Executes a forward step of the basket GRU model, for a given input x
    
    Arguments:
      x               batch of input sequences ([batch size] x [sequence length] x [assortment size])
      fw_dropout      dropout rate to apply in the forward layer (either False (0, default) or a scalar value between 0 and 1)
      rc_dropout      dropout rate to apply in the input gate (either False (0, default) or a scalar value between 0 and 1)
      stepwise        whether to drop out different nodes in each time step (True) or the same nodes in every time step (False, default)
      track_hiddens   whether to return the hidden states computed in each time step (default is False)
    """
    fw_dropout_rate = 0
    rc_dropout_rate = 0

    if track_hiddens:
      hiddens = torch.zeros(x.size()[0],x.size()[1],self.num_hidden)

    if fw_dropout != False:
      fw_dropout_rate = fw_dropout
    
    if rc_dropout != False:
      rc_dropout_rate = rc_dropout
        
    hidden = self.h_init.to(device)
        
    if stepwise == False:
      fw_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.input_dim)-fw_dropout_rate)/(1-fw_dropout_rate)
      fw_dropout_mask_2 =  torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-fw_dropout_rate)/(1-fw_dropout_rate)
      rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)            
        
    pred_sequence = torch.zeros(x.size()[0],x.size()[1],self.input_dim)
    
    for t in np.arange(x.size()[1]):
        if stepwise == True:
            fw_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.input_dim)-fw_dropout_rate)/(1-fw_dropout_rate)
            fw_dropout_mask_2 =  torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-fw_dropout_rate)/(1-fw_dropout_rate)
            rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)    
                    
        xt = x[:,t,:] # xt should have dimensions [batch_size, input_dim]
        
        if fw_dropout != False:
            xt = xt*fw_dropout_mask

        forget_gate = torch.sigmoid(torch.matmul(xt,self.W_fx) + torch.matmul(hidden,self.W_fh) + self.b_f)
        modulator_gate = torch.sigmoid(torch.matmul(xt,self.W_mx) + torch.matmul(hidden,self.W_mh) + self.b_m)
        candidate_gate = torch.tanh(torch.matmul(xt,self.W_cx) + torch.matmul(hidden*modulator_gate,self.W_ch) + self.b_c)
        
        if rc_dropout != False:
            candidate_gate = candidate_gate*rc_dropout_mask
            
        hidden = hidden * (1-forget_gate) + forget_gate*candidate_gate
        
        if fw_dropout != False:
            hidden = hidden * fw_dropout_mask_2

        if track_hiddens:
          hiddens[:,t,:] = hidden

        next_basket = torch.sigmoid(torch.matmul(hidden,self.W_ph) + self.b_p)
        pred_sequence[:,t,:] = next_basket            
        
    if track_hiddens:
      return pred_sequence, hiddens
    else:
      return pred_sequence

class basket_GRUX(nn.Module):
    """
    Class for the basket GRU with covariates, as described by Van Maasakkers, Donkers en Fok (2022)
    """
    
    def __init__(self, batch_size, seq_length, input_dim, num_hidden, seq_dim, hour_dim, seq_hidden, hour_hidden):    
        """
        Initializes a basket_GRUX instance

        Arguments:
          seq_length    number of time steps in each (padded) sequence, scalar
          input_dim     dimensionality of the GRU input (equals the assortment size here)
          num_hidden    dimensionality of the hidden GRU state, scalar hyperparameter
          seq_dim       input dimensionality of the sequence length indicators
          hour_dim      input dimensionality of the hour-of-day dummies
          seq_hidden    dimensionality of the hidden layer in the sequence length indicator branch
          hour_hidden   dimensionality of the hidden layer in the hour-of-day branch
        """
        super(basket_GRUX, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.seq_dim = seq_dim # input dimension of sequential variables
        self.hour_dim = hour_dim # input dimension of hour dummies
        self.seq_hidden = seq_hidden # number of latent variables in hidden seq layer
        self.hour_hidden = hour_hidden # number of factors in hidden hour layer
                
        # Initialize hidden state and cell state
        if (train_init):
            if (train_ind==False):
                self.h_init = nn.Parameter(torch.zeros(num_hidden))
        else:
            self.h_init = torch.zeros(num_hidden)
        
        # Initialize forget gate weights and biases (for input x and hidden state h)
        self.W_fx = nn.Parameter(torch.Tensor(input_dim, num_hidden))
        nn.init.xavier_uniform_(self.W_fx.data)
        self.W_fh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        nn.init.xavier_uniform_(self.W_fh.data)
        self.b_f = nn.Parameter(torch.zeros(num_hidden))
        
        # Initialize input modulator gate weights and biases
        self.W_mx = nn.Parameter(torch.Tensor(input_dim, num_hidden))
        nn.init.xavier_uniform_(self.W_mx.data)
        self.W_mh = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        nn.init.xavier_uniform_(self.W_mh.data)
        self.b_m = nn.Parameter(torch.zeros(num_hidden))
        
        # Initialize input candidate gate weights and biases
        self.W_cx = nn.Parameter(torch.Tensor(input_dim, num_hidden))
        nn.init.xavier_uniform_(self.W_cx.data)
        self.W_ch = nn.Parameter(torch.Tensor(num_hidden, num_hidden))
        nn.init.xavier_uniform_(self.W_ch.data)
        self.b_c = nn.Parameter(torch.zeros(num_hidden))
        
        # Initialize parameters for hour dummy layer        
        self.W_hhour = nn.Parameter(torch.Tensor(hour_dim, hour_hidden))
        nn.init.xavier_uniform_(self.W_hhour.data)
        
        # Initialize parameters for sequence indicator layer
        self.W_hseq = nn.Parameter(torch.Tensor(seq_dim, seq_hidden))
        nn.init.xavier_uniform_(self.W_hseq.data)
        self.b_hseq = nn.Parameter(torch.zeros(seq_hidden))
        
        # Initialize prediction weight and biases
        self.W_ph = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        nn.init.xavier_uniform_(self.W_ph.data)
        self.W_phour = nn.Parameter(torch.Tensor(hour_hidden, input_dim))
        nn.init.xavier_uniform_(self.W_phour.data)
        self.W_pseq = nn.Parameter(torch.Tensor(seq_hidden, input_dim))
        nn.init.xavier_uniform_(self.W_pseq.data)
        self.b_p = nn.Parameter(torch.zeros(input_dim))
        
        
    def forward(self, x, hour, seq, fw_dropout = False, rc_dropout = False, stepwise = False, track_hiddens = False):
        """
        Executes a forward step of the basket GRU model, for a given input x

        Arguments:
          x               batch of input sequences ([batch size] x [sequence length] x [input dim])
          hour            batch of hour-of-day variables ([batch size] x [sequence length] x [hour dim])
          seq             batch of sequence-length indicators ([batch size] x [sequence length] x [seq dim])
          fw_dropout      dropout rate to apply in the forward layer (either False (0, default) or a scalar value between 0 and 1)
          rc_dropout      dropout rate to apply in the input gate (either False (0, default) or a scalar value between 0 and 1)
          stepwise        whether to drop out different nodes in each time step (True) or the same nodes in every time step (False, default)
          track_hiddens   whether to return the hidden states computed in each time step (default is False)
        """

        fw_dropout_rate = 0
        rc_dropout_rate = 0

        if track_hiddens:
          hiddens = torch.zeros(x.size()[0],x.size()[1],self.num_hidden)
        
        if fw_dropout != False:
            fw_dropout_rate = fw_dropout
               
        if rc_dropout != False:
            rc_dropout_rate = rc_dropout
        
        hidden = self.h_init
        
        if stepwise == False:
            fw_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.input_dim)-fw_dropout_rate)/(1-fw_dropout_rate)
            fw_dropout_mask_2 =  torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-fw_dropout_rate)/(1-fw_dropout_rate)
            rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)            
        
        pred_sequence = torch.zeros(x.size()[0],x.size()[1],self.input_dim)
        
        for t in np.arange(x.size()[1]):
            if stepwise == True:
                fw_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.input_dim)-fw_dropout_rate)/(1-fw_dropout_rate)
                fw_dropout_mask_2 =  torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-fw_dropout_rate)/(1-fw_dropout_rate)
                rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)    
                       
            xt = x[:,t,:] # xt should have dimensions [batch_size, input_dim]
            seqt = seq[:,t,:]
            hourt = hour[:,t,:]
            
            if fw_dropout != False:
                xt = xt*fw_dropout_mask
    
            forget_gate = torch.sigmoid(torch.matmul(xt,self.W_fx) + torch.matmul(hidden,self.W_fh) + self.b_f)
            modulator_gate = torch.sigmoid(torch.matmul(xt,self.W_mx) + torch.matmul(hidden,self.W_mh) + self.b_m)
            candidate_gate = torch.tanh(torch.matmul(xt,self.W_cx) + torch.matmul(hidden*modulator_gate,self.W_ch) + self.b_c)
            
            if rc_dropout != False:
                candidate_gate = candidate_gate*rc_dropout_mask
                
            hidden = hidden * (1-forget_gate) + forget_gate*candidate_gate
                        
            if fw_dropout != False:
                hidden = hidden * fw_dropout_mask_2
            
            if track_hiddens:
                hiddens[:,t,:] = hidden
                
            hidden_hour = torch.matmul(hourt,self.W_hhour)
            hidden_seq = nn.functional.relu(torch.matmul(seqt,self.W_hseq) + self.b_hseq)
                        
            next_basket = torch.sigmoid(torch.matmul(hidden,self.W_ph) + torch.matmul(hidden_hour,self.W_phour) + torch.matmul(hidden_seq, self.W_pseq) + self.b_p)
            pred_sequence[:,t,:] = next_basket            
            
        if track_hiddens:
          return pred_sequence, hiddens
        else:
          return pred_sequence

class linear_GRU(nn.Module):
  """
  Class for the linearized version of the basket GRU model
  """
  
  def __init__(self,seq_length,input_dim,num_hidden):
    """
    Initializes a linear_GRU instance
    
    Arguments:
      seq_length    number of time steps in each (padded) sequence, scalar
      input_dim     dimensionality of the GRU input (equals the assortment size here)
      num_hidden    dimensionality of the hidden GRU state, scalar hyperparameter
    """
    super(linear_GRU,self).__init__()
    self.seq_length = seq_length
    self.input_dim = input_dim
    self.num_hidden = num_hidden

    self.h_init = torch.zeros(num_hidden)

    # Weight matrices
    self.Z = nn.Parameter(torch.Tensor(num_hidden,input_dim))
    nn.init.xavier_uniform_(self.Z.data)
    self.b_out = nn.Parameter(torch.zeros(input_dim))

    self.KFinv = nn.Parameter(torch.Tensor(input_dim,num_hidden))
    nn.init.xavier_uniform_(self.KFinv.data)
    self.b_hidden = nn.Parameter(torch.zeros(num_hidden))

  def forward(self, x, fw_dropout = False, rc_dropout = False, stepwise = False, track_hiddens=False):
    """
    Executes a forward step of the basket GRU model, for a given input x

    Arguments:
      x               batch of input sequences ([batch size] x [sequence length] x [input dim])
      fw_dropout      dropout rate to apply in the forward layer (either False (0, default) or a scalar value between 0 and 1)
      rc_dropout      dropout rate to apply in the input gate (either False (0, default) or a scalar value between 0 and 1)
      stepwise        whether to drop out different nodes in each time step (True) or the same nodes in every time step (False, default)
      track_hiddens   whether to return the hidden states computed in each time step (default is False)
    """
    fw_dropout_rate = 0
    rc_dropout_rate = 0

    if track_hiddens:
      hiddens = torch.zeros(x.size()[0],x.size()[1],self.num_hidden)

    if fw_dropout != False:
      fw_dropout_rate = fw_dropout
    
    if rc_dropout != False:
      rc_dropout_rate = rc_dropout
        
    hidden = self.h_init.to(device)
    next_basket = torch.matmul(hidden,self.Z) + self.b_out
        
    if stepwise == False:
      fw_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.input_dim)-fw_dropout_rate)/(1-fw_dropout_rate)
      fw_dropout_mask_2 =  torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-fw_dropout_rate)/(1-fw_dropout_rate)
      rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)            
        
    pred_sequence = torch.zeros(x.size()[0],x.size()[1],self.input_dim)
    
    for t in np.arange(x.size()[1]):
        if stepwise == True:
            fw_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.input_dim)-fw_dropout_rate)/(1-fw_dropout_rate)
            fw_dropout_mask_2 =  torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-fw_dropout_rate)/(1-fw_dropout_rate)
            rc_dropout_mask = torch.bernoulli(torch.ones(x.size()[0],self.num_hidden)-rc_dropout_rate)/(1-rc_dropout_rate)    
                    
        xt = x[:,t,:] # xt should have dimensions [batch_size, input_dim]
        
        if fw_dropout != False:
            xt = xt*fw_dropout_mask
   
        diff = xt - next_basket
        hidden = torch.matmul(diff,self.KFinv) + self.b_hidden

        if track_hiddens:
          hiddens[:,t,:] = hidden
        next_basket = torch.sigmoid(torch.matmul(hidden,self.Z) + self.b_out)
  
        pred_sequence[:,t,:] = next_basket            
        
    if track_hiddens:
      return pred_sequence, hiddens
    else:
      return pred_sequence


class custom_BCE(torch.nn.Module):
  """
  Class for computing the binary cross-entropy loss of model output
  """
  
  def __init__(self, only_last = True):
    """
    Initializes a custom_BCE isntance
    
    Arguments:
      only_last   whether BCE should be computed on the last baskets only, or average over the entire sequence
    """
    self.only_last = only_last
    super(custom_BCE,self).__init__()
  
  def forward(self,pred,target,last):
    """
    Calculates binary cross-entropy loss for model output
     
    Arguments:
      pred    batch of predictions of size [batch_size] x [seq length] x [input dim]
      target  batch of target labels (1 = purchased, 0 = non-purchased) with same size as pred
      last    vector containing the indices of the last baskets of each sequence, with size [batch_size]
    """
    if self.only_last:
        pred_last_basket = pred[torch.arange(target.size()[0]),last-1,:]
        target_last_basket = target[torch.arange(target.size()[0]),last-1,:]
        loss = pred_last_basket.clone()
        loss[target_last_basket==False] = 1 - loss[target_last_basket==False]
    else:
        mask = torch.sum(target,2)>0
        pred_sequence = pred[mask]
        target_sequence = target[mask]
        loss = pred_sequence.clone()
        loss[target_sequence==False] = 1 - loss[target_sequence==False]

    loss[loss<1e-30] = 1e-30
    final_loss = -torch.mean(torch.log(loss))
    return final_loss

class custom_MSE(torch.nn.Module):
  """
    Class to calculate a mean squared error
  """
  
  def __init__(self, only_last = True):
    """
    Initializes a custom_MSE isntance
    
    Arguments:
      only_last   whether MSE should be computed on the last baskets only, or average over the entire sequence
    """
    self.only_last = only_last
    super(custom_MSE,self).__init__()
        
  def forward(self,pred,target,last):
    """
    Calculates mean squared error loss for model output
     
    Arguments:
      pred    batch of predictions of size [batch_size] x [seq length] x [input dim]
      target  batch of target labels (1 = purchased, 0 = non-purchased) with same size as pred
      last    vector containing the indices of the last baskets of each sequence, with size [batch_size]
    """
    if self.only_last:
        pred_last_basket = pred[torch.arange(target.size()[0]),last-1,:]
        target_last_basket = target[torch.arange(target.size()[0]),last-1,:]
        loss = pred_last_basket.clone()
        loss[target_last_basket==True] = 1 - loss[target_last_basket==True]
    else:
        mask = torch.sum(target,2)>0
        pred_sequence = pred[mask]
        target_sequence = target[mask]
        loss = pred_sequence.clone()
        loss[target_sequence==True] = 1 - loss[target_sequence==True]

    final_loss = torch.mean(loss**2)
    return final_loss

def top_prod_acc(pred, target, last):
    """
    Function to calculate the accuracy of the top ranked products by the model
    
    Arguments:
      pred    batch of predictions of size [batch_size] x [seq length] x [input dim]
      target  batch of target labels (1 = purchased, 0 = non-purchased) with same size as pred
      last    vector containing the indices of the last baskets of each sequence, with size [batch_size]
    """
    pred = pred.cpu()
    target = target.cpu()
    last = last.cpu()
    pred_last_basket = pred[torch.arange(target.size()[0]),last-1,:]
    target_last_basket = target[torch.arange(target.size()[0]),last-1,:]
    target_items = target_last_basket.nonzero() 
    pred_ranking = torch.sort(pred_last_basket,dim=1,descending=True).indices
    acc = 0
    for i in np.arange(0,pred.size()[0]):
        target_prods = target_items[target_items[:,0]==i,1]
        pred_prods = pred_ranking[i,0:target_prods.size()[0]]
        diff = np.setdiff1d(target_prods.numpy(),pred_prods.numpy())
        acc += 1-(np.shape(diff)[0]/target_prods.size()[0])
    av_acc = acc/pred.size()[0]
    return av_acc   

def batch_generator(input_seq, output_seq, index_last, batch_size, num_users, shuffle = True):
    """
    Batch generator for model without covariates
    
    Arguments:
      input_seq     all input sequences with size [number of users] x [sequence length] x [assortment size]
      output_seq    all target sequences with size [number of users] x [sequence length] x [assortment size]
      index_last    vector containing the indices of the last baskets of each sequence, with size [number of users]
      batch_size    size of the batches provided to the model in each iteration
      num_users     total number of users
      shuffle       whether to randomly shuffle sequences or not once new batches are created (True by default)
    """
    number_of_batches = num_users/batch_size
    counter = 0
    index = np.arange(np.shape(input_seq)[0])
    if shuffle:
        np.random.shuffle(index)
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        input_batch = input_seq[index_batch,:,:]
        output_batch = output_seq[index_batch,:,:]
        index_last_batch = index_last[index_batch]
        counter += 1
        yield(index_batch, input_batch,output_batch,index_last_batch)
        if (counter >= number_of_batches):
            if shuffle:
                np.random.shuffle(index)
            counter=0

def batch_generatorX(input_seq, hour, seq, output_seq, index_last, batch_size, num_users, shuffle = True):
    """
    Batch generator for model with covariates
    
    Arguments:
      input_seq     all input sequences with size [number of users] x [sequence length] x [assortment size]
      hour          all hour-of-day dummies with size [number of users] x [sequence length] x [hour dim]
      seq           all sequence length indicators with size [number of users] x [sequence length] x [seq dim]
      output_seq    all target sequences with size [number of users] x [sequence length] x [assortment size]
      index_last    vector containing the indices of the last baskets of each sequence, with size [number of users]
      batch_size    size of the batches provided to the model in each iteration
      num_users     total number of users
      shuffle       whether to randomly shuffle sequences or not once new batches are created (True by default)
    """
    number_of_batches = num_users/batch_size
    counter = 0
    index = np.arange(np.shape(input_seq)[0])
    if shuffle:
        np.random.shuffle(index)
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        input_batch = input_seq[index_batch,:,:]
        hour_batch = hour[index_batch,:,:]
        seq_batch = seq[index_batch,:,:]
        output_batch = output_seq[index_batch,:,:]
        index_last_batch = index_last[index_batch]
        counter += 1
        yield(index_batch, input_batch,hour_batch,seq_batch,output_batch,index_last_batch)
        if (counter >= number_of_batches):
            if shuffle:
                np.random.shuffle(index)
            counter=0            

def BCE(pred,target):
    """
    Function to calculate binary cross-entorpy with numpy inputs
    
    Arguments:
      pred    set of input baskets with size [num baskets] x [assortment size]
      target  set of target baskets (1 = purchased, 0 = non-purchased) with same size as pred
      last    vector containing the indices of the last baskets of each sequence, with size [num baskets]
    """
    loss = np.copy(pred)
    loss[target==0.0] = 1 - loss[target==0.0]
    loss[loss<1e-50] = 1e-50
    final_loss = -np.mean(np.log(loss))
    return final_loss

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
  """
  Function used to quickly read csv files
  """
  def iter_func():
      with open(filename, 'r') as infile:
          for _ in range(skiprows):
              next(infile)
          for line in infile:
              line = line.rstrip().split(delimiter)
              for item in line:
                  yield dtype(item)
      iter_loadtxt.rowlength = len(line)

  data = np.fromiter(iter_func(), dtype=dtype)
  data = data.reshape((-1, iter_loadtxt.rowlength))
  return data

