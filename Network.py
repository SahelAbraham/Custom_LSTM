#A custom made LSTM, so I can understand how they work!
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

class LSTM(L.LightningModule):
  
  def __init__(self):
    #Initializes the LSTM weights and biases
    pass

  def lstm_unit(self, input, long_memory, short_memory):
    #Does all the lstm math
    pass

  def forward(self, input):
    #Moves forward through the network
    pass

  def optimizer_config(self):
    #Configures the optimizer
    pass

  def training(self, batch, batch_ind):
    #Calculates the loss values and training progression
    pass