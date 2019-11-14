#python plot_cifar10.py --batch-size 128 --dataset cifar10 --noise_type pairflip --noise_rate 0.45 --n_epoch 100 --eps 9.99 --lr 0.001

import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
#from model import CNN
import argparse, sys
import numpy as np
import datetime
import shutil
#from net_10 import Net
import PreResNet
import matplotlib.pyplot as plt


'''class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        #self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        #self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 20)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x#F.log_softmax(x, dim=1)
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train(args, model, device, train_loader, optimizer, epoch, eps=9.9):
    model.train()

    loss_t=[]
    loss_c=[]
    for batch_idx, (data, target,index, kind) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output)
        loss = F.nll_loss(output, target,reduction='none')
        kind = torch.squeeze(kind,1)
        #print(kind)
        #print(loss.shape,kind.shape,kind)
        l1=loss[kind==True].clone()
        l2=loss[kind==False].clone()
        #print(loss.shape,l1.shape,l2.shape)
        loss_t.extend(l1.detach().cpu().numpy())
        loss_c.extend(l2.detach().cpu().numpy())
        loss = torch.sum(loss)/len(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    #print(loss_t,loss_c)
    plt.figure()
    plt.hist([loss_t,loss_c], bins=50, range=[0,5], density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=['red','blue'], label=["true","false"], stacked=True)
    #plt.hist(loss_c, bins=30, range=[0,4], density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color='blue', label="false", stacked=False, normed=False)
    plt.legend()
    plt.savefig("img/distribition_"+str(epoch)+".jpg",dpi=400)
    plt.close()
    return loss.item()#, torch.mean(output[:,10])

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    acc = []
    with torch.no_grad():
        for data, target, index, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output[:,:10].argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc=100. * correct / len(test_loader.dataset);
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return acc
    
    
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
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
    parser.add_argument('--num_iter_per_epoch', type=int, default=400)
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument('--eps', type=float, default=9.9)
    
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=4000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    batch_size=args.batch_size
    
    if args.dataset=='mnist':
        input_channel=1
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = 200
        train_dataset = MNIST(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                             )

        test_dataset = MNIST(root='./data/',
                                   download=True,  
                                   train=False, 
                                   transform=transforms.ToTensor(),
                                   noise_type=args.noise_type,
                                   noise_rate=args.noise_rate
                            )
    
    if args.dataset=='cifar10':
        input_channel=3
        num_classes=10
        args.top_bn = False
        args.epoch_decay_start = 80
        args.n_epoch = 200
        train_dataset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                               )

        test_dataset = CIFAR10(root='./data/',
                                    download=True,  
                                    train=False, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                              )

    if args.dataset=='cifar100':
        input_channel=3
        num_classes=100
        args.top_bn = False
        args.epoch_decay_start = 100
        args.n_epoch = 200
        train_dataset = CIFAR100(root='./data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )

        test_dataset = CIFAR100(root='./data/',
                                    download=True,  
                                    train=False, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )

    if args.forget_rate is None:
        forget_rate=args.noise_rate
    else:
        forget_rate=args.forget_rate

    noise_or_not = train_dataset.noise_or_not
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    #print('building model...')
    #cnn = CNN(input_channel=input_channel, n_outputs=num_classes*2)
    #cnn = Net().to(device)
    #cnn.cuda()
    #print(cnn1.parameters)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cnn = PreResNet.ResNet18(num_classes=10).to(device)
    #cnn = Net().to(device)
    cnn.cuda()
    #print(model.parameters)
    #optimizer1 = torch.optim.SGD(cnn1.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
    

    acc=[]
    loss=[]
    loss_pure=[]
    loss_corrupt=[]
    out=[]
    for epoch in range(1, args.n_epoch + 1):
        if epoch<20:
            l1=train(args, cnn, device, train_loader, optimizer, epoch,eps=args.eps)
            loss.append(l1)
            #out.append(out10)
            acc.append(test(args, cnn, device, test_loader))
        else:
            l1=train(args, cnn, device, train_loader, optimizer, epoch,eps=args.eps)
            loss.append(l1)
            #out.append(out10)
            acc.append(test(args, cnn, device, test_loader))
        
    name=str(args.dataset)+" "+str(args.noise_type)+" "+str(args.noise_rate)

if __name__=='__main__':
    main()
