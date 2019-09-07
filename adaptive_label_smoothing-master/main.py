# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from model import CNN
import argparse, sys
import numpy as np
import datetime
import shutil

def train(args, model, device, train_loader, optimizer, epoch, eps=9.9):
    model.train()
   # for batch_idx, (data, target, idx, is_pure, is_corrupt) in enumerate(train_loader):
    for batch_idx, (data, target,index) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.softmax(output, dim=1)
        output = output + (output[:, 10] / eps).unsqueeze(1)
        output.log_()
        
        loss = F.nll_loss(output, target)
        
        loss.backward()
        
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    acc = []
    with torch.no_grad():
        for data, target, index in test_loader:
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
    parser.add_argument('--eps', type=float, default=9.3)

    
    parser.add_argument('--batch-size', type=int, default=600, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=4000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
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
    print('building model...')
    cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes+1)
    cnn1.cuda()
    print(cnn1.parameters)
    #optimizer1 = torch.optim.SGD(cnn1.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(cnn1.parameters(), lr=args.lr, momentum=args.momentum)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    save_dir = args.result_dir +'/' +args.dataset+'/new_loss/'

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    model_str=args.dataset+'_new_loss_'+args.noise_type+'_'+str(args.noise_rate)+'_'+str(args.eps)+' '
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    txtfile=save_dir+"/"+model_str+nowTime+".txt"
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))
    acc=[]
    loss=[]
    loss_pure=[]
    loss_corrupt=[]
    for epoch in range(1, args.n_epoch + 1):
#         if epoch<20:
#             loss.append(train(args, model, device, train_loader, optimizer, epoch, eps=9.5))
#         else:
#             loss.append(train(args, model, device, train_loader, optimizer, epoch, eps=5))
        #loss.append(train(args, model, device, train_loader, optimizer, epoch, eps=9.3))
        l1=train(args, cnn1, device, train_loader, optimizer, epoch, eps=args.eps)
        loss.append(l1)
        acc.append(test(args, cnn1, device, test_loader))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(acc[-1]) + "\n")

    name=str(args.dataset)+" "+str(args.noise_type)+" "+str(args.noise_rate)
    np.save(name+"acc.npy",acc)
    np.save(name+"loss.npy",loss)
    
if __name__=='__main__':
    main()