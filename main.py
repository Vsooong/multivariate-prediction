__author__ = "Guan Song Wang"

import argparse
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import LSTNet,MHA_Net,CNN,RNN
import importlib

from utils import *
from train_eval import train, evaluate, makeOptimizer

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, required=True,help='location of the data file')
parser.add_argument('--model', type=str, default='LSTNet', help='')
parser.add_argument('--window', type=int, default=24 * 7,help='window size')
parser.add_argument('--horizon', type=int, default=12)

parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units each layer')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of RNN hidden layers')

parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units (channels)')
parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)

parser.add_argument('--clip', type=float, default=10.,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')

parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--amsgrad', type=str, default=True)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')
args = parser.parse_args()



# Choose device: cpu or gpu
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility.
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
Data = Data_utility(args.data, 0.6, 0.2, device, args)
# loss function
if args.L1Loss:
    criterion = nn.L1Loss(size_average=False)
else:
    criterion = nn.MSELoss(size_average=False)
evaluateL2 = nn.MSELoss(size_average=False)
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

# Select model
model = eval(args.model).Model(args, Data)
train_method = train
eval_method = evaluate
nParams = sum([p.nelement() for p in model.parameters()])
print('number of parameters: %d' % nParams)
if args.cuda:
    model = nn.DataParallel(model)


best_val = 10000000

optim = makeOptimizer(model.parameters(), args)

# While training you can press Ctrl + C to stop it.
try:
    print('Training start')
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss = train_method(Data, Data.train[0], Data.train[1], model, criterion, optim, args)

        val_loss, val_rae, val_corr = eval_method(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args)
        print('| end of epoch {:3d} | time used: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.
                format( epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))

        if val_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 10 == 0:
            test_acc, test_rae, test_corr = eval_method(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args)
            print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}\n".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args)
print('Best model performance：')
print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
