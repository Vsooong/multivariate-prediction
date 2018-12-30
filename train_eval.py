import math
import torch.optim as optim
import torch
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
        optimizer = optim.SGD(params, lr=args.lr,)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr,)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr,)
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr,)
    else:
        raise RuntimeError("Invalid optim method: " + args.method)
    return optimizer

# class Optim(object):
#
#     def _makeOptimizer(self):
#         if self.method == 'sgd':
#             self.optimizer = optim.SGD(self.params, lr=self.lr)
#         elif self.method == 'adagrad':
#             self.optimizer = optim.Adagrad(self.params, lr=self.lr)
#         elif self.method == 'adadelta':
#             self.optimizer = optim.Adadelta(self.params, lr=self.lr)
#         elif self.method == 'adam':
#             self.optimizer = optim.Adam(self.params, lr=self.lr)
#         else:
#             raise RuntimeError("Invalid optim method: " + self.method)
#
#     def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None):
#         self.params = list(params)  # careful: params may be a generator
#         self.last_ppl = None
#         self.lr = lr
#         self.max_grad_norm = max_grad_norm
#         self.method = method
#         self.lr_decay = lr_decay
#         self.start_decay_at = start_decay_at
#         self.start_decay = False
#
#         self._makeOptimizer()
#
#     def step(self):
#         # Compute gradients norm.
#         grad_norm = 0
#         for param in self.params:
#             grad_norm += math.pow(param.grad.data.norm(), 2)
#
#         grad_norm = math.sqrt(grad_norm)
#         if grad_norm > 0:
#             shrinkage = self.max_grad_norm / grad_norm
#         else:
#             shrinkage = 1.
#
#         for param in self.params:
#             if shrinkage < 1:
#                 param.grad.data.mul_(shrinkage)
#
#         self.optimizer.step()
#         return grad_norm
#
#     # decay learning rate if val perf does not improve or we hit the start_decay_at limit
#     def updateLearningRate(self, ppl, epoch):
#         if self.start_decay_at is not None and epoch >= self.start_decay_at:
#             self.start_decay = True
#         if self.last_ppl is not None and ppl > self.last_ppl:
#             self.start_decay = True
#
#         if self.start_decay:
#             self.lr = self.lr * self.lr_decay
#             print("Decaying learning rate to %g" % self.lr)
#         #only decay for one epoch
#         self.start_decay = False
#
#         self.last_ppl = ppl
#
#         self._makeOptimizer()
