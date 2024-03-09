import torch
from torch import nn

class RLNetwork(nn.Module):
  def __init__(self, sizes):
      super(RLNetwork, self).__init__()
      self.layers = nn.ModuleList([nn.Linear(sizes[i - 1], sizes[i]) for i in range(1, len(sizes))])


  def forward(self, x):
    curr_x = x
    for layer in self.layers[:-1]:
      curr_x = layer(curr_x)
      curr_x = nn.functional.relu(curr_x)
    return nn.functional.softmax(self.layers[-1](curr_x))  # Softmax for probability distribution
