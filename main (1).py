import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()




import math
import torch
from torch.optim import Optimizer


class LaProp(Optimizer):
    def __init__(self, params, lr=4e-4, betas=(0.8, 0.99), eps=1e-15,
                 weight_decay=0, amsgrad=False, centered=False, Nesterov=1E20):
        #betas = (betas, 1 - (1 - betas) ** 2)
        self.centered = centered
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(LaProp, self).__init__(params, defaults)
        self.Nesterov=Nesterov

    def __setstate__(self, state):
        super(LaProp, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        Nesterov = False
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)#.uniform_( -2 *group['eps'], 2 * group['eps'])
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)#.uniform_(group['eps'] ** 2 , (2 * group['eps']) ** 2)
                    state['exp_mean_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_mean_avg_sq = state['exp_avg'], state['exp_avg_sq'], state['exp_mean_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                #if self.Nesterov:
                Nesterov = ((state['step'] % 2) % self.Nesterov == self.Nesterov - 1)
                #print()

                if self.centered:
                    exp_mean_avg_sq.mul_(beta2).add_(1 - beta2, grad)
                    mean = exp_mean_avg_sq ** 2
                bias_correction1 = 1 - beta1 ** state['step']
                if Nesterov == True:
                    bias_correction1 = 1 - beta1 ** (state['step'] + 1)
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    print('No!')
                else:

                    
                    denom = exp_avg_sq
                    if self.centered:
                        denom = denom - mean
                    denom = (denom / bias_correction2).sqrt().add(group['eps'])
                    grad2 = grad / denom

                    exp_avg.mul_(beta1).add_(1 - beta1, grad2)
                    if Nesterov == True:
                        advance_m2 = exp_avg_sq.mul(beta2).add(1 - beta2, grad **2).sqrt().add(group['eps'])
                        bias_correction2 = 1 - beta2 ** (state['step'] + 1)
                        grad2 = grad / (advance_m2 / math.sqrt(bias_correction2))
                        advance_m = exp_avg.mul(beta1).add(1 - beta1, grad2)
                
                
                step_size = group['lr'] / bias_correction1

                #if state['step'] >= 50:
                if True:
                    if Nesterov:
                        p.data.add_(-step_size, advance_m)
                    else:
                        p.data.add_(-step_size, exp_avg)
                    if group['weight_decay'] != 0:
                        p.data.add_(group['lr'] * group['weight_decay'], - p.data)
        if state['step'] == 50:
            print('start')
        return 0, 0, 0, 0

def clip_gradient(model, clip_value):
    return
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if torch.cuda.is_available():
        model.cuda()
    #optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optim = LaProp(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        #print(prediction.shape)
        
        eps = 1.999
        output = F.softmax(prediction, dim=1)
        output = output + (output[:, 2] / eps).unsqueeze(1)
        output.log_()
        loss = F.nll_loss(output, target)


        num_corrects = (torch.max(prediction[:, :2], 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print ('Epoch:', epoch+1, 'Idx:', idx+1, 'Training Loss:', loss.item(), 'Training Accuracy:', acc.item())
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)

            #eps = 9.9
            #output = F.softmax(prediction, dim=1)
            #output = output + (prediction[:, 2] / eps).unsqueeze(1)
            #output.l()
            loss = F.nll_loss(prediction, target)

            num_corrects = (torch.max(prediction[:, :2], 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
    

learning_rate = 1e-4
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300

model = LSTMClassifier(batch_size, output_size+1, hidden_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy

for epoch in range(100):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc = eval_model(model, valid_iter)
    
    print('Epoch:', epoch+1, 'Train Loss:', train_loss, 'Train Acc:', train_acc, 'Val. Loss:', val_loss, 'Val. Acc:', val_acc)
    
test_loss, test_acc = eval_model(model, test_iter)
print('Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

assert False

''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
test_tensor = Variable(test_sen, volatile=True)
test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Sentiment: Positive")
else:
    print ("Sentiment: Negative")
