import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
#from data.cifar import CIFAR10, CIFAR100
#from data.mnist import MNIST
#from model import CNN

#from torchlist import ImageFilelist
from clothing import Clothing
import argparse, sys
import numpy as np
import datetime
import shutil
#from net_10 import Net
import ResNet
import matplotlib.pyplot as plt
import scipy.stats as stats
from torchvision import datasets


def reg_loss_class(mean_tab,num_classes=10):
    loss = 0
    for items in mean_tab:
        loss += (1./num_classes)*torch.log((1./num_classes)/items)
    return loss


######################### Get data and noise adding ##########################
def get_data_cifar(loader):
    data = loader.sampler.data_source.data.copy()
    labels = loader.sampler.data_source.targets
    labels = torch.Tensor(labels[:]).long() # this is to copy the list
    return (data, labels)

def get_data_cifar_2(loader):
    #print(type(loader.sampler.data_source.targets))
    labels = loader.sampler.data_source.targets
    #print(labels)
    labels = torch.Tensor(labels[:]).long() # this is to copy the list
    return labels

#Noise without the sample class
def add_noise_cifar_wo(loader, noise_percentage = 20,noise_type = 'pairflip'):
    torch.manual_seed(2)
    np.random.seed(42)
    if noise_type == 'symmetric':
        print(noise_type)
        noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
        #images = [sample_i for sample_i in loader.sampler.data_source.data]
        probs_to_change = torch.randint(100, (len(noisy_labels),))
        idx_to_change = probs_to_change >= (100.0 - noise_percentage)
        percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

        for n, label_i in enumerate(noisy_labels):
            if idx_to_change[n] == 1:
                set_labels = list(
                    set(range(10)) - set([label_i]))  # this is a set with the available labels (without the current label)
                set_index = np.random.randint(len(set_labels))
                noisy_labels[n] = set_labels[set_index]

        #loader.sampler.data_source.data = images
        loader.sampler.data_source.targets = noisy_labels
    elif noise_type == 'pairflip':
        print(noise_type)
        torch.manual_seed(2)
        np.random.seed(42)
        noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
        #images = [sample_i for sample_i in loader.sampler.data_source.data]
        probs_to_change = torch.randint(100, (len(noisy_labels),))
        idx_to_change = probs_to_change >= (100.0 - noise_percentage)
        percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

        for n, label_i in enumerate(noisy_labels):
            if idx_to_change[n] == 1:
                #set_labels = list(set(range(10)))  # this is a set with the available labels (with the current label)
                #set_index = np.random.randint(len(set_labels))
                t_labels = noisy_labels[n]
                noisy_labels[n] = (t_labels+1) % 10#set_labels[set_index]

        #loader.sampler.data_source.data = images
        loader.sampler.data_source.targets = noisy_labels
    return noisy_labels

#Noise with the sample class (as in Re-thinking generalization )
def add_noise_cifar_w(loader, noise_percentage = 20, noise_type = 'pairflip'):
    if noise_type == 'symmetric':
        print(noise_type)
        torch.manual_seed(2)
        np.random.seed(42)
        noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
        images = [sample_i for sample_i in loader.sampler.data_source.data]
        probs_to_change = torch.randint(100, (len(noisy_labels),))
        idx_to_change = probs_to_change >= (100.0 - noise_percentage)
        percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

        for n, label_i in enumerate(noisy_labels):
            if idx_to_change[n] == 1:
                set_labels = list(set(range(10)))  # this is a set with the available labels (with the current label)
                set_index = np.random.randint(len(set_labels))
                noisy_labels[n] = set_labels[set_index]

        loader.sampler.data_source.data = images
        loader.sampler.data_source.targets = noisy_labels
    elif noise_type == 'pairflip':
        print(noise_type)
        torch.manual_seed(2)
        np.random.seed(42)
        noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
        images = [sample_i for sample_i in loader.sampler.data_source.data]
        probs_to_change = torch.randint(100, (len(noisy_labels),))
        idx_to_change = probs_to_change >= (100.0 - noise_percentage)
        percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

        for n, label_i in enumerate(noisy_labels):
            if idx_to_change[n] == 1:
                #set_labels = list(set(range(10)))  # this is a set with the available labels (with the current label)
                #set_index = np.random.randint(len(set_labels))
                t_labels = noisy_labels[n]
                noisy_labels[n] = (t_labels+1) % 10#set_labels[set_index]

        loader.sampler.data_source.data = images
        loader.sampler.data_source.targets = noisy_labels
    return noisy_labels


def compute_probabilities_batch(data, target, cnn_model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    cnn_model.eval()
    out_t = cnn_model(data)
    outputs = F.log_softmax(out_t, dim=1)
    batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
    batch_losses.detach_()
    outputs.detach_()
    cnn_model.train()
    batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
    batch_losses[batch_losses >= 1] = 1-10e-4
    batch_losses[batch_losses <= 0] = 10e-4

    #B = bmm_model.posterior(batch_losses,1)
    B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

    return torch.FloatTensor(B)

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def track_training_loss(args, model, device, train_loader, epoch, bmm_model1, bmm_model_maxLoss1, bmm_model_minLoss1):
    model.eval()

    all_losses = torch.Tensor()
    all_predictions = torch.Tensor()
    all_probs = torch.Tensor()
    all_argmaxXentropy = torch.Tensor()

    #for batch_idx, (data, target,index, kind) in enumerate(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)

        prediction = F.log_softmax(prediction, dim=1)
        idx_loss = F.nll_loss(prediction, target, reduction = 'none')
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
        probs = prediction.clone()
        probs.detach_()
        all_probs = torch.cat((all_probs, probs.cpu()))
        arg_entr = torch.max(prediction, dim=1)[1]
        arg_entr = F.nll_loss(prediction.float(), arg_entr.to(device), reduction='none')
        arg_entr.detach_()
        all_argmaxXentropy = torch.cat((all_argmaxXentropy, arg_entr.cpu()))

    loss_tr = all_losses.data.numpy()

    # outliers detection
    max_perc = np.percentile(loss_tr, 95)
    min_perc = np.percentile(loss_tr, 5)
    loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)]

    bmm_model_maxLoss = torch.FloatTensor([max_perc]).to(device)
    bmm_model_minLoss = torch.FloatTensor([min_perc]).to(device) + 10e-6


    loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)

    loss_tr[loss_tr>=1] = 1-10e-4
    loss_tr[loss_tr <= 0] = 10e-4

    bmm_model = BetaMixture1D(max_iters=10)
    bmm_model.fit(loss_tr)

    bmm_model.create_lookup(1)

    return all_losses.data.numpy(), \
           all_probs.data.numpy(), \
           all_argmaxXentropy.numpy(), \
           bmm_model, bmm_model_maxLoss, bmm_model_minLoss


def train_CE(args, model, device, train_loader, optimizer, epoch,num_classes):
    model.train()
    #for batch_idx, (data, target,index, kind) in enumerate(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(type(data),type(target))
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out_t =  model(data)
        #out_f = out_f.detach()
        #out_f,out_t = F.log_softmax(out_t,dim=1),F.log_softmax(out_t,dim=1)
        
        out_t = F.log_softmax(out_t[:num_classes],dim=1)
        loss = F.nll_loss(out_t,target)
        #loss = 0.00*F.nll_loss(out_f, target,reduction="none")+(1)*F.nll_loss(out_t,target,reduction="none")
        #loss = torch.sum(loss)/len(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

def train(args, model, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss,temp,num_classes):
    model.train()
    #for batch_idx, (data, target,index, kind) in enumerate(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out_t = model(data)
        #output = F.log_softmax(output,dim=1)
        #loss = F.nll_loss(output, target,reduction='none')
        B = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
        B = B.to(device)
        B[B <= 1e-4] = 1e-4
        B[B >= 1 - 1e-4] = 1 - 1e-4
        #print(B.shape)
        #print(B)
        out_t = F.softmax(out_t,dim=1)
        #print(B.shape,out_f.shape,out_t.shape)
        weight = torch.from_numpy(np.array([B.cpu().numpy() for i in range(num_classes)],dtype='f'))
        weight = torch.transpose(weight,0,1).cuda()
        weight = weight/(1-weight)
        weight[weight >= 1e3] = 1e3
        weight[weight <= 1e-4] = 1e-4
        
        #output = torch.tensor([128,10])
        #for i in range(128):
        #    k = out_t[i]*(1-B[i])+out_f[i]*B[i]
        #    print(k.shape,output[i].shape)
        #    output[i]=k
        #print(weight.shape,out_t.shape)
        #output = (1-weight) * out_t +( weight) * out_f*temp
        output = out_t[:,:num_classes].clone +(weight) * out_t[:,num_classes:].clone#*temp
        output = torch.log(output+0.0001)
        output = torch.log_softmax(output,dim=1)
        loss = F.nll_loss(output,target)   #B*F.nll_loss(out_f, target,reduction="none")+(1-B)*F.nll_loss(out_t,target,reduction="none")
        #loss = #torch.sum(loss)/len(loss)
        loss.backward()
        optimizer.step()
        #temp -= 0.0025
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print("temp=",temp," classes=",num_classes)

    return loss.item(),temp#, torch.mean(output[:,10])


def train_true_label(args, model, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    model.train()
    #for batch_idx, (data, target,index, kind) in enumerate(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        feature, out_f, out_t = model(data)
        #out_f,out_t =F.softmax(out_f,dim=1), F.softmax(out_t,dim=1)
        out_f = out_f.detach()
        #print(out_f.requires_grad)
        output = (out_f+out_t)
        output = F.log_softmax(output,dim=1)
        #output = torch.log(output+0.00001)
        loss = F.nll_loss(output,target)
        #out_t_mean = F.softmax(out_t, dim=1)
        #out_t = F.log_softmax(out_f,dim=1)
        #tab_mean_class = torch.mean(out_t_mean,-2)
        #loss_reg = reg_loss_class(tab_mean_class, 10)
        #print(out_t.argmax(dim=1).shape)
        #loss = F.nll_loss(out_t,out_t.argmax(dim=1)) # + loss_reg*0.1
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()  # , torch.mean(output[:,10])

def train_uncertainty(args, model, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    model.train()
    #for batch_idx, (data, target,index, kind) in enumerate(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        feature, out_f, out_t = model(data)
        #out_t_mean = F.softmax(out_t, dim=1)
        #out_f,out_t =F.softmax(out_f,dim=1), F.softmax(out_t,dim=1)
        #tab_mean_class = torch.mean(out_t_mean,-2)
        #loss_reg = reg_loss_class(tab_mean_class, 10)
        out_t = out_t.detach()
        #print(out_t.requires_grad)
        output = (out_f+out_t)# + loss_reg
        #print(torch.sum(output))
        #h = torch.sum(output,dim=1)
        #print(h)
        #output = torch.log(output+0.00001)
        output = F.log_softmax(output,dim=1)
        loss = F.nll_loss(output,target)# + loss_reg
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()  # , torch.mean(output[:,10])




def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    #print(outputs>1)
    #outputs[outputs <= 1e-4] = 1e-4
    #outputs[outputs >= 1 - 1e-4] = 1 - 1e-4

    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), 10).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = y_onehot @ outputs
    #print(mat.float())
    #print(torch.abs(torch.det(mat.float())))
    #loss = -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)
    #print(loss)
    return  -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.000001)

def train_DMI(args, model, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss, criterion):
    model.train()
    #for batch_idx, (data, target,index, kind) in enumerate(train_loader):
    for batch_idx, ( input, target) in enumerate(train_loader):
        if input.size(0) != args.batch_size:
            continue

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())
      
        _, _, output = model(input)
        loss = criterion(output, target)
        #loss = DMI_loss(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    '''
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        feature, out_f, out_t = model(data)
        #out_f,out_t =F.softmax(out_f,dim=1), F.softmax(out_t,dim=1)
        #output = (out_f+out_t)/2
        output=F.softmax(out_t,dim=1)
        loss = DMI_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    '''
    return loss.item()  # , torch.mean(output[:,10])

def train_together(args, model, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        feature, out_f, out_t = model(data)
        out_t_mean = F.softmax(out_t, dim=1)
        #out_f, out_t = F.log_softmax(out_t, dim=1), F.log_softmax(out_f, dim=1)
        out = F.log_softmax(out_t)
        out_f, out_t = F.softmax(out_t, dim=1), F.softmax(out_f, dim=1)
        output = (out_f+out_t)/2
        #loss_t = F.nll_loss((out_t + out_f) / 2, target)
        loss_t = F.cross_entropy(output,target)
        tab_mean_class = torch.mean(out_t_mean,-2)
        loss_reg = reg_loss_class(tab_mean_class, 10)
        loss_DMI = DMI_loss(output, target)
        loss = F.cross_entropy(out_t,out_t.argmax(dim=1))#F.nll_loss(out_t,out_t.argmax(dim=1))# + loss_reg*0.1 + loss_t + DMI_loss((out_t + out_f) / 2, target)
        #print(loss_DMI)
        loss = loss + loss_reg*0.1 + loss_t# + loss_DMI
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    return loss.item()  # , torch.mean(output[:,10])

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    acc = []
    with torch.no_grad():
        for data,target in test_loader:#for data, target, index, kind in test_loader:
            data, target = data.to(device), target.to(device)
            #_,out1,out2=model(data)
            #test_loss += F.nll_loss(F.softmax(out2,dim=1), target, reduction='sum').item()
            _,_,output = model(data)
            test_loss += F.nll_loss(F.softmax(output,dim=1), target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc=100. * correct / len(test_loader.dataset);
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return acc

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
    parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
    parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
    parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='symmetric')
    parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    parser.add_argument('--top_bn', action='store_true')
    parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'mnist')
    parser.add_argument('--n_epoch', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=2, help='how many subprocesses to use for data loading')
    parser.add_argument('--num_iter_per_epoch', type=int, default=400)
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument('--eps', type=float, default=9.9)
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=4000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                        help='learning rate (default: 0.01)')   #sm 0.5   lr=0.5
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--noise-level', type=float, default=80.0,
                        help='percentage of noise added to the data (values from 0. to 100.), default: 80.')
    parser.add_argument('--root-dir', type=str, default='.', help='path to CIFAR dir where cifar-10-batches-py/ and cifar-100-python/ are located. If the datasets are not downloaded, they will automatically be and extracted to this path, default: .')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    batch_size=args.batch_size
    
    # if args.dataset=='mnist':
    #     input_channel=1
    #     num_classes=10
    #     args.top_bn = False
    #     args.epoch_decay_start = 80
    #     args.n_epoch = 200
    #     train_dataset = MNIST(root='./data/',
    #                                 download=True,
    #                                 train=True,
    #                                 transform=transforms.ToTensor(),
    #                                 noise_type=args.noise_type,
    #                                 noise_rate=args.noise_rate
    #                          )
    #
    #     test_dataset = MNIST(root='./data/',
    #                                download=True,
    #                                train=False,
    #                                transform=transforms.ToTensor(),
    #                                noise_type=args.noise_type,
    #                                noise_rate=args.noise_rate
    #                         )
    #
    # if args.dataset=='cifar10':
    #     input_channel=3
    #     num_classes=10
    #     args.top_bn = False
    #     args.epoch_decay_start = 80
    #     args.n_epoch = 200
    #     train_dataset = CIFAR10(root='./data/',
    #                                 download=True,
    #                                 train=True,
    #                                 transform=transforms.ToTensor(),
    #                                 noise_type=args.noise_type,
    #                                 noise_rate=args.noise_rate
    #                            )
    #
    #     test_dataset = CIFAR10(root='./data/',
    #                                 download=True,
    #                                 train=False,
    #                                 transform=transforms.ToTensor(),
    #                                 noise_type=args.noise_type,
    #                                 noise_rate=args.noise_rate
    #                           )
    #
    # if args.dataset=='cifar100':
    #     input_channel=3
    #     num_classes=100
    #     args.top_bn = False
    #     args.epoch_decay_start = 100
    #     args.n_epoch = 200
    #     train_dataset = CIFAR100(root='./data/',
    #                                 download=True,
    #                                 train=True,
    #                                 transform=transforms.ToTensor(),
    #                                 noise_type=args.noise_type,
    #                                 noise_rate=args.noise_rate
    #                             )
    #
    #     test_dataset = CIFAR100(root='./data/',
    #                                 download=True,
    #                                 train=False,
    #                                 transform=transforms.ToTensor(),
    #                                 noise_type=args.noise_type,
    #                                 noise_rate=args.noise_rate
    #                             )
    # if args.forget_rate is None:
    #     forget_rate=args.noise_rate
    # else:
    #     forget_rate=args.forget_rate
    #
    # noise_or_not = train_dataset.noise_or_not
    # # Data Loader (Input Pipeline)
    # print('loading dataset...')
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=batch_size,
    #                                            num_workers=args.num_workers,
    #                                            drop_last=True,
    #                                            shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=batch_size,
    #                                           num_workers=args.num_workers,
    #                                           drop_last=True,
    #                                           shuffle=False)
    '''
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])'''
    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = datasets.CIFAR10(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset_track = datasets.CIFAR10(root=args.root_dir, train=True, transform=transform_train)
        testset = datasets.CIFAR10(root=args.root_dir, train=False, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        trainset = datasets.CIFAR100(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset_track = datasets.CIFAR100(root=args.root_dir, train=True, transform=transform_train)
        testset = datasets.CIFAR100(root=args.root_dir, train=False, transform=transform_test)
        num_classes = 100
    elif args.dataset == 'imagenet_tiny':
        init_epoch = 100
        num_classes = 200
        #data_root = '/home/xingyu/Data/phd/data/imagenet-tiny/tiny-imagenet-200'
        data_root = '/home/iedl/w00536717/coteaching_plus-master/data/imagenet-tiny/tiny-imagenet-200'
        train_kv = "train_noisy_%s_%s_kv_list.txt" % (args.noise_type, args.noise_rate)
        test_kv = "val_kv_list.txt"

        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std =[0.2302, 0.2265, 0.2262])

        trainset = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv),
                   transform=transforms.Compose([transforms.RandomResizedCrop(56),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normalize,
           ]))
        trainset_track = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv),
                   transform=transforms.Compose([transforms.RandomResizedCrop(56),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normalize,
           ]))

        testset = ImageFilelist(root=data_root, flist=os.path.join(data_root, test_kv),
                   transform=transforms.Compose([transforms.Resize(64),
                   transforms.CenterCrop(56),
                   transforms.ToTensor(),
                   normalize,
           ]))
    elif args.dataset == 'clothing1M':
        init_epoch =100
        num_classes = 14
        data_root = '/home/iedl/w00536717/data/cloting1m/'
        #train_kv =
        #test_kv =
         
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
        '''
        train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ]) 
        '''
              
    ''' 
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    train_loader_track = torch.utils.data.DataLoader(trainset_track, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)
    '''
    train_dataset = Clothing(root=data_root, img_transform=train_transform, train=True, valid=False, test=False)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 32)
    train_dataset_track = Clothing(root=data_root, img_transform=train_transform, train=True, valid=False, test=False)
    train_loader_track = torch.utils.data.DataLoader(dataset = train_dataset_track, batch_size = args.batch_size, shuffle = False, num_workers = 32)
    valid_dataset = Clothing(root=data_root, img_transform=train_transform, train=False, valid=True, test=False)
    valid_loader = torch.utils.data.DataLoader(dataset =valid_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 32)
    test_dataset = Clothing(root=data_root, img_transform=test_transform, train=False, valid=False, test=True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 32)
    #print(dir(train_loader))
    #labels = get_data_cifar_2(train_loader_track)  # it should be "clonning"
    #noisy_labels = add_noise_cifar_wo(train_loader, args.noise_level,
    #                                 args.noise_type)  # it changes the labels in the train loader directly
    #noisy_labels_track = add_noise_cifar_wo(train_loader_track, args.noise_level, args.noise_type)

    # Define models
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #cnn = PreResNet_two.ResNet18(num_classes=num_classes).to(device)
    cnn = MyResNet_zero.MyCustomResnet(num_classes)#.to(device)
    cnn = nn.DataParallel(cnn,device_ids=[0,1,2,3,4,5,6,7])
    cnn.to(device)
    cnn.cuda()
    #print(model.parameters)
    #optimizer1 = torch.optim.SGD(cnn1.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr,weight_decay=1e-3,momentum=args.momentum)
    #optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr,weight_decay=1e-4)
    #optimizer1 = torch.optim.SGD(cnn.parameters(), lr=1e-2,weight_decay=1e-4,momentum=args.momentum)
    bmm_model = bmm_model_maxLoss = bmm_model_minLoss=0

    acc=[]
    loss=[]
    loss_pure=[]
    loss_corrupt=[]
    out=[]
    temp=1
    for epoch in range(1, args.n_epoch + 1):
        if epoch<3:
            #epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
            #    track_training_loss(args, cnn, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss,
            #                        bmm_model_minLoss)
            #l1,temp=train(args, cnn, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss,temp)
            l1=train_CE(args, cnn, device, train_loader, optimizer, epoch)
            #adjust_learning_rate(optimizer, 0.1)
            #l2=train_uncertainty(args, cnn, device, train_loader, optimizer1, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            #l1=train_true_label(args, cnn, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            #l2=train_uncertainty(args, cnn, device, train_loader, optimizer1, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            #l1=train_CE(args, cnn, device, train_loader, optimizer, epoch)
            #l2=train_uncertainty(args, cnn, device, train_loader, optimizer1, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            #loss.append(l1)
            acc.append(test(args, cnn, device, test_loader))
        elif epoch<80:
            if epoch==3:
                adjust_learning_rate(optimizer, args.lr/10)
            epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
                track_training_loss(args, cnn, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss,
                                    bmm_model_minLoss)
            l1,temp=train(args, cnn, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss,temp,num_classes)
            #l1=train_CE(args, cnn, device, train_loader, optimizer, epoch)
            #l2=train_uncertainty(args, cnn, device, train_loader, optimizer1, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            acc.append(test(args, cnn, device, test_loader))
        elif epoch<200:
            if epoch==10:
                adjust_learning_rate(optimizer, args.lr/1000)
            elif epoch==100:
                adjust_learning_rate(optimizer, args.lr/5000)
            elif epoch==160:
                adjust_learning_rate(optimizer, args.lr/25000)
            epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
                track_training_loss(args, cnn, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss,
                                    bmm_model_minLoss)

            l1,temp=train(args, cnn, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss,temp,num_classes)
            #adjust_learning_rate(optimizer, args.lr/1000)
            #epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
            #    track_training_loss(args, cnn, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss,
            #                        bmm_model_minLoss)
            #l2=train_uncertainty(args, cnn, device, train_loader, optimizer1, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            #acc.append(test(args, cnn, device, test_loader))
            #l1=train_true_label(args, cnn, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            #l2=train_uncertainty(args, cnn, device, train_loader, optimizer1, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            #l3=train_DMI(args, cnn, device, train_loader, optimizer1, epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, criterion = DMI_loss)
            #l1 = train_together(args, cnn, device, train_loader, optimizer, epoch, bmm_model, bmm_model_maxLoss,
            #                      bmm_model_minLoss)
            #loss.append(l1)
            #out.append(out10)
            acc.append(test(args, cnn, device, test_loader))
        else:
            #adjust_learning_rate(optimizer, args.lr/10000)
            epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
                track_training_loss(args, cnn, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss,
                                    bmm_model_minLoss)
            l1,temp=train(args, cnn, device, train_loader, optimizer, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss,temp,num_classes)
            #l2=train_uncertainty(args, cnn, device, train_loader, optimizer1, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            #acc.append(test(args, cnn, device, test_loader))
            #l1 = train_true_label(args, cnn, device, train_loader, optimizer, epoch, bmm_model, bmm_model_maxLoss,bmm_model_minLoss)
            #l2 = train_uncertainty(args, cnn, device, train_loader, optimizer1, epoch, bmm_model, bmm_model_maxLoss,bmm_model_minLoss)
            #l3=train_DMI(args, cnn, device, train_loader, optimizer1, epoch,bmm_model, bmm_model_maxLoss, bmm_model_minLoss, criterion = DMI_loss)
            #l1 = train_together(args, cnn, device, train_loader, optimizer, epoch, bmm_model, bmm_model_maxLoss,
            #                    bmm_model_minLoss)
            #loss.append(l1)
            #out.append(out10)
            acc.append(test(args, cnn, device, test_loader))
        print("Evaluate on validset")
        test(args, cnn, device, valid_loader)
        temp -= 0.0024
        
    name=str(args.dataset)+" "+str(args.noise_type)+" "+str(args.noise_rate)

if __name__=='__main__':
    main()
