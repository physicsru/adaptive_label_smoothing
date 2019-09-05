from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from net import Net
import random
import sys
import numpy as np


class MyDataset(Dataset):
    def __init__(self,rate):
        self.mnist = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))
        self.ran=[i for i in range(0,60000)]
        random.seed(42)
        torch.manual_seed(0)
        self.index=random.sample(self.ran,int(60000*rate))
        self.index_yes=list(set(range(60000))-set(self.index))
        
    def __getitem__(self, index):
        data, target = self.mnist[index]
        is_pure = index in self.index_yes
        is_corrupt = index in self.index
        return data, target, index, is_pure, is_corrupt

    def __len__(self):
        return len(self.mnist)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, idx, is_pure, is_corrupt) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss_pure = F.nll_loss(output[is_pure],target[is_pure])
        loss_corrupt = F.nll_loss(output[is_corrupt],target[is_corrupt])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss, loss_pure, loss_corrupt
        
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss_pure = F.nll_loss(output[:,index_yes],target[:,index_yes])
#         loss_corrupt = F.nll_loss(output[:,index],target[:,index])
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
        #print(loss.item())
    #return loss.item()

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    acc = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc=100. * correct / len(test_loader.dataset);
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=600, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=4000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
     
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         x,
#         batch_size=args.batch_size, shuffle=True, **kwargs)
    dataset = MyDataset(0.8)
    loader = DataLoader(dataset,
                        batch_size=600,
                        shuffle=True,
                        num_workers=10)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
#     train_loader=x

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    acc=[]
    loss=[]
    loss_pure=[]
    loss_corrupt=[]
    for epoch in range(1, args.epochs + 1):
        l1,l2,l3=train(args, model, device, loader, optimizer, epoch)
        loss.append(l1)
        loss_pure.append(l2)
        loss_corrupt.append(l3)
        acc.append(test(args, model, device, test_loader))
    np.save("acc.npy",acc)
    np.save("loss.npy",loss)
    np.save("loss_pure.npy",loss_pure)
    np.save("loss_corrupt.npy",loss_corrupt)
    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

main()
