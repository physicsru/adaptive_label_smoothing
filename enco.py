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

class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        #self.vgg = models.vgg19_bn(pretrained=True)
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.ModuleList(resnet50.children())[:-1]
        self.features = nn.Sequential(*self.features)
        #self.vgg_features = self.vgg.features
        #self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        #features = self.fc_features(features)
        return features
