__author__ = "Guan Song Wang"

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.window = args.window
        self.variables = data.m
        self.hw=args.highway_window
        self.activate1=F.relu
        self.hidR=args.hidRNN
        self.rnn1=nn.GRU(self.variables,self.hidR,num_layers=args.rnn_layers)
        self.linear1 = nn.Linear(self.hidR, self.variables)
        # self.linear1=nn.Linear(1280,100)
        # self.out=nn.Linear(100,self.variables)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)

        self.dropout = nn.Dropout(p=args.dropout)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        r= x.permute(1,0,2).contiguous()
        _,r=self.rnn1(r)
        r=self.dropout(torch.squeeze(r[-1:,:,:], 0))
        out = self.linear1(r)


        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.variables)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out
