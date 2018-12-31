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

        self.d_v=args.d_v
        self.d_k=args.d_k
        self.Ck = args.CNN_kernel
        self.GRU1=nn.GRU(self.variables,self.hidR,num_layers=args.rnn_layer)
        self.Conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.variables))

        self.slf_attn = MultiHeadAttention(args.n_head, self.hidR, self.d_k,self.d_v , dropout=args.dropout)

        # self.pooling=nn.MaxPool2d(kernel_size=(self.window,1))
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear_out=nn.Linear(self.window*self.hidR,self.variables)

        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh


    def forward(self, x):
        batch_s=x.size(0)
        r = x.permute(1, 0, 2).contiguous()
        out, _ = self.GRU1(r)
        c = out.permute(1, 0, 2).contiguous()

        # c = x.view(-1,1, self.window, self.variables)
        #
        # c = F.relu(self.Conv1(c))
        # c = self.dropout(c)
        # c = torch.squeeze(c, 3)
        # c=c.permute(0,2,1).contiguous()

        attn_output, slf_attn=self.slf_attn(c,c,c,mask=None)

        # attn_output=attn_output.view(-1,1,self.window,self.d_k)
        #
        # pool=self.pooling(attn_output).squeeze()
        # pool=self.dropout(pool)

        # print(attn_output.size())

        attn_output=attn_output.view(batch_s,-1)
        res=self.linear_out(attn_output)

        if (self.output):
            res = self.output(res)
        return res