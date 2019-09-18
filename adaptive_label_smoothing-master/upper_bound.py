from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
#from net import Net
import random
import sys
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.accuracy=[]
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 800)
        #x = x.view(x.size(0), x.size(1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) 
    
class MyDataset(Dataset):
    def __init__(self,rate):
        self.mnist = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))
        self.ran=[i for i in range(0,60000)]
        rate = float(rate)
        #print(type(rate))
        #random.seed(42)
        #torch.manual_seed(0)
        #print(rate)
        #print(60000*rate)
        self.index=random.sample(self.ran,int(60000*rate))
        self.index_yes=list(set(range(60000))-set(self.index))
        self.mnist.train_labels[self.index]=self.mnist.train_labels[self.index].random_(0,10)
        self.mnist=[self.mnist[i] for i in self.index_yes]
        
    def __getitem__(self, index):
        data, target = self.mnist[index]
        is_pure = index in self.index_yes
        is_corrupt = index in self.index
        return data, target, index, is_pure, is_corrupt

    def __len__(self):
        return len(self.mnist)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    loss1=[]
    loss2=[]
    loss3=[]
    
    for batch_idx, (data, target, idx, is_pure, is_corrupt) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss_pure = F.nll_loss(output[is_pure==1],target[is_pure==1])
        loss_corrupt = F.nll_loss(output[is_corrupt==1],target[is_corrupt==1])
        loss.backward()
        optimizer.step()
        
        loss1.append(loss.item())
        loss2.append(loss_pure.item())
        loss3.append(loss_corrupt.item())
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item(), loss_pure.item(), loss_corrupt.item()

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
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--noise_rate', default=0.5,
                        help='noise rate')
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
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
#     train_loader=x
    noise_rate = args.noise_rate
    #print(noise_rate)
    dataset = MyDataset(noise_rate)
    loader = DataLoader(dataset,
                        batch_size=128,
                        shuffle=True,
                        num_workers=4)
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
    np.save("upper_bound "+str(noise_rate)+" "+str(args.seed)+" acc.npy",acc)
    np.save("upper_bound "+str(noise_rate)+" "+str(args.seed)+" loss.npy",loss)
    #np.save("loss_pure_p.npy",loss_pure)
    #np.save("loss_corrupt_p.npy",loss_corrupt)
    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn_p.pt")

if __name__=='__main__':
    main()
