__author__ = "Guan Song Wang"

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.window = args.window
        self.variables = data.m

        self.conv1 = nn.Conv1d(self.variables, 32, kernel_size=3)
        self.activate1=F.relu
        self.conv2=nn.Conv1d(32,32,kernel_size=3)
        self.maxpool1=nn.MaxPool1d(kernel_size=2)
        self.conv3=nn.Conv1d(32,16,kernel_size=3)
        self.maxpool2=nn.MaxPool1d(kernel_size=2)
        self.linear1=nn.Linear(1280,100)
        self.out=nn.Linear(100,self.variables)

        self.dropout = nn.Dropout(p=args.dropout)


    def forward(self, x):
        c = x.permute(0,2,1).contiguous()
        x=self.conv1(c)
        x=self.activate1(x)
        x=self.conv2(x)
        x=self.activate1(x)
        x=self.maxpool1(x)
        x=self.conv3(x)
        x=self.activate1(x)
        x=x.view(x.size(0),x.size(1)*x.size(2))
        x=self.dropout(x)
        x=self.linear1(x)
        x=self.dropout(x)
        x=self.out(x)
        return x
