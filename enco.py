import torch
from torchvision.datasets import ImageFolder
import model
from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((500, 375)),
    transforms.ToTensor()

])

dataset=ImageFolder('/home/super/code/DSCMR/dataset', transform= transform)
size = len(dataset)
#print(size)
batch_size = 16
train_set, test_set = torch.utils.data.random_split(dataset, [size//5 * 4, size//5])
#print(len(train_set),len(test_set))
train_loader = torch.utils.data.DataLoader(
	train_set,
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
	test_set,
	batch_size=batch_size, shuffle=False)
device = torch.device("cuda")
model = model.VGGNet().to(device)
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    print(data.shape,target.shape)
    output = model(data)
    print(output)

