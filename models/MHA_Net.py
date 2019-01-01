__author__ = "Guan Song Wang"

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MultiHeadAttention import MultiHeadAttention

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.window = args.window
        self.variables = data.m
        self.hidC = args.hidCNN
        self.hidR = args.hidRNN
        self.hw=args.highway_window

        self.d_v=args.d_v
        self.d_k=args.d_k
        self.Ck = args.CNN_kernel
        self.GRU = nn.GRU(self.variables, self.hidR, num_layers=args.rnn_layers)
        # self.Conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.variables))

        self.slf_attn = MultiHeadAttention(args.n_head, self.variables, self.d_k,self.d_v , dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.hidR,self.variables)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh


    def forward(self, x):
        # r = x.permute(1, 0, 2).contiguous()
        # out, _ = self.GRU1(r)
        # c = out.permute(1, 0, 2).contiguous()
        # c = x.view(-1,1, self.window, self.variables)
        # c = F.relu(self.Conv1(c))
        # c = self.dropout(c)
        # c = torch.squeeze(c, 3)
        # c=c.permute(0,2,1).contiguous()

        attn_output, slf_attn=self.slf_attn(x,x,x,mask=None)

        r=attn_output.permute(1,0,2).contiguous()
        _,r=self.GRU(r)
        r = self.dropout(torch.squeeze(r[-1:, :, :], 0))
        out = self.linear_out(r)

        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.variables)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out