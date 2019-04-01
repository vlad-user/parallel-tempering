import torch
import torch.nn as nn

class Lenet5(nn.Module):
  def __init__(self, dropout_rate=1):
    super(Lenet5, self).__init__()
    self.dropout_rate = dropout_rate
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3,
                  out_channels=6,
                  kernel_size=5,
                  padding=0,
                  bias=True),
        nn.ReLU(),
        nn.MaxPool2d(2))
    nn.init.xavier_uniform_(self.conv1[0].weight)
    nn.init.zeros_(self.conv1[0].bias)
    
    self.dropout1 = nn.Dropout(p=dropout_rate)
    
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=6,
                  out_channels=12,
                  kernel_size=3,
                  bias=True),
        nn.ReLU(),
        nn.MaxPool2d(2))
    nn.init.xavier_uniform_(self.conv2[0].weight)
    nn.init.zeros_(self.conv2[0].bias)
    
    self.dropout2 = nn.Dropout(p=dropout_rate)
    
    self.logits = nn.Linear(432, 10)
    nn.init.xavier_uniform_(self.logits.weight)
    nn.init.zeros_(self.logits.bias)


  def get_n_trainable_params(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  def forward(self, x):
    x = self.conv1(x)
    x = self.dropout1(x)
    x = self.conv2(x)
    x = self.dropout2(x)
    x = x.view(x.size(0), -1)
    x = self.logits(x)
    return x

