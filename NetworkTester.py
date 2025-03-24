#Testing our LSTM using the concept of 3 different company stocks
from Network import LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint

model = LSTM()

#Initialize our training data to be used
inputs = torch.tensor([[0.0, 0.5, 0.25, 1], [1.0, 0.5, 0.25, 1], [0.5, 0.5, 0.25, 1]])
labels = torch.tensor([0.0, 1.0, 0.5])
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

trainer = L.Trainer(max_epochs=2000)
trainer.fit(model, train_dataloaders=dataloader)

path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path


#Company A goes from [0.0, 0.5, 0.25, 1, 0]
print(f"Company A: Observed = 0, Predicted = {model(torch.tensor([0.0, 0.5, 0.25, 1]).detach())}")

#Company B goes from [1.0, 0.5, 0.25, 1, 1]
print(f"Company B: Observed = 1, Predicted = {model(torch.tensor([1.0, 0.5, 0.25, 1]).detach())}")

#Company C goes from [0.5, 0.5, 0.25, 1, 0.5]
print(f"Company C: Observed = 0.5, Predicted = {model(torch.tensor([0.5, 0.5, 0.25, 1]).detach())}")

