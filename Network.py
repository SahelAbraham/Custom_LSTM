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
    super.__init__()
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)

    self.w1s1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
    self.w2s1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
    self.bs1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    self.w1s2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
    self.w2s2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
    self.bs2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    self.w1s3 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
    self.w2s3 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
    self.bs3 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    self.w1s4 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
    self.w2s4 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
    self.bs4 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

  def lstm_unit(self, input, long_mem, short_mem):
    #Does all the lstm math

    #The percent of the long memory we want to keep
    long_memory_percent = torch.sigmoid(short_mem * self.w1s1 + input * self.w2s2 + self.bs1)
    #The percentage of the potential long term memory we want to add to a percent of the current long term memory
    potential_mem_percent = torch.sigmoid(short_mem * self.w1s2 + input * self.w2s2 + self.bs2)
    #The actual potential long term memory that we will add a percent of to a percent of the current long term memory
    potential_memory = torch.tanh(short_mem * self.w1s3 + input * self.w2s3 + self.bs3)

    #We get the new long term memory by adding a percent of the current long term memory to a percent of the potential long term memory
    updated_long_mem = (long_mem * long_memory_percent) + (potential_mem_percent * potential_memory)

    #the percent of the short term memory we want to keep
    output_percent = torch.sigmoid(short_mem * self.w1s4 + input * self.w2s4 + self.bs4)

    #The new short term memory is the tanh of the new long term memory times the percent of long term memory we want to keep as short term memory
    updated_short_mem = torch.tanh(updated_long_mem) * output_percent

    return [updated_long_mem, updated_short_mem]

  def forward(self, input):
    #Moves forward through the network
    pass

  def optimizer_config(self):
    #Configures the optimizer
    pass

  def training(self, batch, batch_ind):
    #Calculates the loss values and training progression
    pass