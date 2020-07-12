import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--bs', default=16, type=int, help='batch_size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda'# if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = CIFAR10(
    root='./data', train=True, download=True, transform=transform_train, noise_type="clean")
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.bs, shuffle=False, num_workers=16, drop_last=True)

testset = CIFAR10(
    root='./data', train=False, download=True, transform=transform_test, noise_type="clean")
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=16)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = RegNetX_200MF()
#net = ResNet34()
net = net.to(device)
n_class = 10


#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True

def proportion_compute(label_input):
    batch_size = len(label_input)
    #print(label_input.T.shape)
    #print(torch.transpose(torch.unsqueeze(label_input,1), 0, 1).shape)
    #print(torch.unsqueeze(label_input, 1).shape)
    y_onehot = torch.zeros(batch_size, n_class).scatter_(1, torch.unsqueeze(label_input, 1), 1)
    prob_result = y_onehot.sum(axis = 0)/batch_size
    #print(label_input)
    #print(prob_result)
    return  prob_result

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=5e-4)

def llploss(output, prop):
    #D_loss_prop = tf.reduce_mean(-tf.reduce_sum(y_prob * (tf.log(tf.reduce_mean(tf.nn.softmax(d_net_real), [0]) + 1e-7))))
    return
def zerooneloss():
    return

def CE():
    return

# Training
def TrainFloatingLoss(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, index) in enumerate(trainloader):
        y_prob_mb = proportion_compute(targets)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #loss = torch.sum(outputs, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def TrainPropLoss(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, index) in enumerate(trainloader):
        y_prob_mb = proportion_compute(targets)
        y_prob_mb = y_prob_mb.to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        #print(y_prob_mb, torch.sum(y_prob_mb))
        optimizer.zero_grad()
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        #outputs = -torch.log_softmax(torch.sum(outputs, dim = 0) / len(index)+ 1e-7, dim = 0)
        #print(torch.sum(outputs, dim=0).shape, F.softmax(outputs, dim = 1).shape)
        outputs = -torch.log(torch.sum(F.softmax(outputs, dim = 1), dim=0) / len(index) + 1e-7)
        #print(outputs.shape, y_prob_mb.shape)
        loss = torch.dot(y_prob_mb, outputs)  #criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+3000):
    TrainPropLoss(epoch)
    test(epoch)
