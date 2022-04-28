from functools import partial
import torch
from torch import nn

    
class GRU(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout=0., win_len=1024, number_of_direction=36):
        super(GRU,self).__init__()
        freqs = (win_len // 2 + 1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.number_of_direction = number_of_direction
        self.num_of_frames = 386

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(21,5), stride=1, padding=0)
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(21,5), stride=1, padding=0)
        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(21,5), stride=1, padding=0)
        self.maxpool = nn.MaxPool2d((5, 2), padding=(2, 0))
        self.time_rnn = nn.GRU(1092, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.time_fc1 = nn.Linear(hidden_size, hidden_size)
        self.time_fc2 = nn.Linear(hidden_size, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x, states=None):
        """
        Args:
            x: [B, T, F, C] - mic & num of direction
            states (tuple): each [num_layers, B, hidden]
        """
        B, T, F, C = x.shape
        if states is None:
            states = x.new_zeros((self.num_layers, B, self.hidden_size))

        x = x.reshape(B*T, 1, F, C)  # [B*T, F, C]
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.maxpool(x)

        x = x.reshape(B, T, -1)  # [B, T, F*C]
        time_rnn_out, states = self.time_rnn(x, states)  # [B, T, H]
        
        out = self.time_fc1(time_rnn_out)  # [B, T, F]
        out = torch.relu(out)
        out = self.time_fc2(out)
        mask = self.activation(out)

        return mask 
       
