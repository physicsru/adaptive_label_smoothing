import json
import torch
from torchvision.datasets import ImageFolder,DatasetFolder
import model
from torchvision import transforms
from sentence_transformers import SentenceTransformer

from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class Pascal(ImageFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(Pascal, self).__init__(root, 
                                    transform=transform,
                                    target_transform=target_transform
                                    )
        self.imgs = self.samples
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        txt = path.split('/')
        txtpath = ""
        for i in txt:
            if i == "dataset":
                txtpath += '/'+ 'sentence'
            else:
                txtpath += '/'+ i
        txtpath = txtpath[1:]
        txtpath = txtpath[:-3] + 'txt'
        f = open(txtpath, "r")
        #print(f.readline())
        txtout = f.readline().replace('\n', '')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, txtout
    
  transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([1.3746831, 1.3277708, 1.2313741], [0.69170624, 0.6828435,  0.70520234])
])

path = '/home/hammer/DSCMR-master/dataset'
dataset=Pascal(path, transform= transform)

means = [0, 0, 0]
stdevs = [0, 0, 0]
import numpy as np
for i, j , k in dataset:
    img = i
    for i in range(3):
        means[i] += img[i, :, :].mean()
        stdevs[i] += img[i, :, :].std()

means = np.asarray(means) / len(dataset)
stdevs = np.asarray(stdevs) / len(dataset)
bn = 128
train_loader = torch.utils.data.DataLoader(dataset, batch_size=bn, shuffle=False)
device = torch.device("cuda")
torch.cuda.set_device(3)
model_img = model.VGGNet().to(device)
model_img.eval()
embedding_img = np.zeros((len(dataset), 2048))
label_txt = np.zeros(len(dataset))
for batch_idx, (i,j,k) in enumerate(train_loader):
    i = i.to(device)
    output = model_img(i)
    embedding_img[batch_idx*bn : batch_idx*bn + output.shape[0]] = output.cpu().detach().numpy()
    label_txt[batch_idx*bn : batch_idx*bn + output.shape[0]] = j
    print(batch_idx*bn, batch_idx*bn + output.shape[0])
np.save("pascal_embedding_img.npy", embedding_img)
np.save( "pascal_label_txt.npy", label_txt)
device = torch.device("cuda")
torch.cuda.set_device(3)
model_txt = SentenceTransformer('bert-large-nli-stsb-mean-tokens').to(device)
model_txt.eval()
embedding_txt = np.zeros((len(dataset), 1024))
for batch_idx, (i,j,k) in enumerate(train_loader):
    output = model_txt.encode(k)
    print(k)
    embedding_txt[batch_idx*bn : batch_idx*bn + output.shape[0]] = output#.cpu().detach().numpy()
    print(batch_idx*bn, batch_idx*bn + output.shape[0])
 np.save("pascal_embedding_txt.npy", embedding_txt)
