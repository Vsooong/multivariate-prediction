import math
import torch.optim as optim
import torch
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, args):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, args.batch_size, False):
        output = model(X)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).data.item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).data.item()
        n_samples += (output.size(0) * data.m)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation


def train(data, X, Y, model, criterion, optim, args):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, args.batch_size, True):
        optim.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optim.step()
        total_loss += loss.data.item()
        n_samples += (output.size(0) * data.m)
    return total_loss / n_samples


def makeOptimizer(params, args):
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, )
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr, )
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr, )
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, )
    else:
        raise RuntimeError("Invalid optim method: " + args.method)
    return optimizer

