#author Guan Song Wang
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.window = args.window
        self.variables = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN

        self.GRU1=nn.GRU(self.variables,self.hidR,num_layers=args.rnn_layer)
        self.Conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.window, 1))
        self.dropout = nn.Dropout(p=args.dropout)
        


    def forward(self, x):

        batch_size = x.size(0)

        r=x.permute(1,0,2).contiguous()
        out,_=self.GRU1(r)

        c=out.permute(1,0,2).contiguous()
        c=c.view(-1,1,self.window, c.size(2))
        c=self.Conv1(c).squeeze(2)
        print(c.size())

        return
        

        # CNN
        c = x.view(-1, 1, self.window, self.variables)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)

        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if (self.skip > 0):
            self.pt=int(self.pt)
            s = c[:, :, int(-self.pt * self.skip):].contiguous()

            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)


        # highway
        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.variables)
            res = res + z

        if (self.output):
            res = self.output(res)
        return res