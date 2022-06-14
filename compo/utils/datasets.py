from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, DatasetFolder
import os

class BasicDataset(Dataset):
    def __init__(self, data, target, transform, target_transform):
        super().__init__()
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target is not None:
            target = self.target[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            return index, img, target
        else:
            return index, img
    
    def __len__(self):
        return len(self.data)

class CIFAR(Dataset):
    def __init__(self, name, data_dir, train=True, transform=None, target_transform=None, download=False, labeled=True):
        self.name = name
        if name == 'cifar10':
            obj = CIFAR10
        elif name == 'cifar100':
            obj = CIFAR100
        else:
            exit(f'dataset {name} is not supported')

        self.transform = transform
        self.target_transform = target_transform
        self.labeled = labeled

        root = os.path.join(data_dir, name)
        cifar_dataobj = obj(root, train, transform, target_transform, download)
        self.data = np.array(cifar_dataobj.data)
        self.target = np.array(cifar_dataobj.targets)
    
    def index_labels(self):
        label_dict = {}
        for i in range(len(self.target)):
            label_dict.setdefault(self.target[i], []).append(i)
        return label_dict
    
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.labeled:
            return index, img, target
        else:
            return index, img

    def __len__(self):
        return len(self.data)

def trunc_ds(ds, idxs, labeled=True):
    # if 'cinic' in ds.name:
    #     data, target = [], []
    #     for i in idxs:
    #         _, img, label = ds[i]
    #         data.append(img)
    #         target.append(label)
    # else:
    data = ds.data[idxs]
    target = ds.target[idxs] if labeled else None

    return BasicDataset(data, target, ds.transform, ds.target_transform)

class Imgs(Dataset):
    def __init__(self, name, data_dir, train=True, transform=None, target_transform=None, labeled=True):
        self.name = name
        if name == 'cinic10':
            fdir = 'train' if train else 'test'
            root = os.path.join(data_dir, 'cinic10', fdir)
        else:
            exit(f'dataset {name} is not supported')

        self.ds = ImageFolder(root=root, transform=transform)
        self.transform = None
        self.target_transform = target_transform
        self.labeled = labeled
        

    def index_labels(self):
        label_dict = {}
        for i in range(len(self.ds)):
            _, target = self.ds[i]
            label_dict.setdefault(int(target), []).append(i)
        return label_dict
    
    def __getitem__(self, index):
        img, target = self.ds[index]

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.labeled:
            return index, img, target
        else:
            return index, img

    def __len__(self):
        return len(self.ds)

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, labeled=True):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.labeled = labeled

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.labeled:
            return index, sample, target
        else:
            return index, sample

    def index_labels(self):
        label_dict = {}
        for i in range(len(self.samples)):
            target = self.samples[i][1]
            label_dict.setdefault(int(target), []).append(i)
        return label_dict

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
